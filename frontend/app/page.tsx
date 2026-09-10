"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import axios from "axios";
import {
  Play,
  Search,
  Film,
  Clock,
  Edit2,
  Check,
  X,
  AlertCircle,
  Plus,
  ArrowRight,
  FolderOpen,
  Filter,
  Type,
  FileVideo,
  FileJson,
  Download,
  ChevronRight,
  ChevronDown,
  Upload,
  Folder,
  File as FileIcon,
  ChevronLeft,
  Home,
  History,
  RotateCcw,
  Undo2,
  Combine,
  Trash2,
  Scissors
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";

interface Segment {
  phase: string;
  start_time: string;
  end_time: string;
  orig_start_time: string; // Persistent original time
  orig_end_time: string;   // Persistent original time
  duration: number;
  video_id: string;
  video_path: string | null;
  instrument: string;
  num_instruments: number;
}

interface FSItem {
  name: string;
  path: string;
  is_dir: boolean;
  size?: number;
}

const API_BASE_URL = "http://127.0.0.1:8000";

const PHASE_COLORS: Record<string, string> = {
  "Diagnosis": "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  "Anesthesia": "bg-sky-500/20 text-sky-400 border-sky-500/30",
  "Incision": "bg-rose-500/20 text-rose-400 border-rose-500/30",
  "Suturing": "bg-amber-500/20 text-amber-400 border-amber-500/30",
  "Cleaning": "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
  "Local injection": "bg-purple-500/20 text-purple-400 border-purple-500/30",
  "Hemostasis": "bg-rose-600/20 text-rose-300 border-rose-500/30",
  "Incision / Dissection": "bg-orange-500/20 text-orange-400 border-orange-500/30",
  "default": "bg-zinc-500/20 text-zinc-400 border-zinc-500/30"
};

const PRESET_PHASES = ["Diagnosis", "Local injection", "Hemostasis", "Incision / Dissection"];

const getPhaseColor = (phase: string) => {
  // Preserve original core brand colors
  if (PHASE_COLORS[phase]) return PHASE_COLORS[phase];

  // Generate stable color based on string hash for new phases
  let hash = 0;
  for (let i = 0; i < phase.length; i++) {
    hash = phase.charCodeAt(i) + ((hash << 5) - hash);
  }

  const h = Math.abs(hash % 360);
  const s = 60 + (Math.abs(hash) % 20); // 60-80% saturation
  const l = 40 + (Math.abs(hash) % 15); // 40-55% lightness

  return `bg-[hsl(${h},${s}%,${l}%,0.2)] text-[hsl(${h},${s}%,${l + 20}%)] border-[hsl(${h},${s}%,${l}%,0.3)]`;
};

const timeToSeconds = (timeStr: string) => {
  if (!timeStr) return 0;
  const parts = timeStr.split(':');
  if (parts.length !== 3) return 0;
  return parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseInt(parts[2]);
};

const secondsToTime = (seconds: number) => {
  const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
  const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${h}:${m}:${s}`;
};

export default function ProVideoDashboard() {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [history, setHistory] = useState<Segment[][]>([]); // Undo History Stack
  const [csvPath, setCsvPath] = useState("");
  const [folderPath, setFolderPath] = useState("");
  const [importInputs, setImportInputs] = useState({ csv: "", folder: "", projectName: "" });
  const [projectName, setProjectName] = useState("PhasePro Project");
  const [projectHistory, setProjectHistory] = useState<{ csv: string, folder: string, name: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentSegment, setCurrentSegment] = useState<Segment | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPhaseFilter, setSelectedPhaseFilter] = useState<string | null>(null);
  const [showEditedOnly, setShowEditedOnly] = useState(false);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [segmentsBackup, setSegmentsBackup] = useState<Segment[] | null>(null);
  const [editValues, setEditValues] = useState<{ start: number, end: number, duration: number, phase: string }>({ start: 0, end: 0, duration: 0, phase: "" });
  const [showImport, setShowImport] = useState(true);

  // Load history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('phasepro_history');
    if (saved) {
      try {
        setProjectHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history", e);
      }
    }
  }, []);

  const saveToHistory = (csv: string, folder: string, name: string) => {
    setProjectHistory(prev => {
      // Check if this exact CSV/Folder pair already exists
      const existingIdx = prev.findIndex(h => h.csv === csv && h.folder === folder);
      let newHistory = [...prev];

      if (existingIdx !== -1) {
        // If it exists, move it to the top (and update name if changed)
        const entry = { ...newHistory[existingIdx], name };
        newHistory.splice(existingIdx, 1);
        newHistory = [entry, ...newHistory];
      } else {
        // Check for name duplicates to add suffix
        let finalName = name;
        let suffix = 1;
        while (newHistory.some(h => h.name === finalName)) {
          finalName = `${name}_${suffix}`;
          suffix++;
        }
        newHistory = [{ csv, folder, name: finalName }, ...newHistory];
      }

      const limitedHistory = newHistory.slice(0, 50);
      localStorage.setItem('phasepro_history', JSON.stringify(limitedHistory));
      return limitedHistory;
    });
  };

  const deleteFromHistory = (e: React.MouseEvent, csv: string, folder: string) => {
    e.stopPropagation(); // Don't trigger loading the project
    setProjectHistory(prev => {
      const newHistory = prev.filter(h => !(h.csv === csv && h.folder === folder));
      localStorage.setItem('phasepro_history', JSON.stringify(newHistory));
      return newHistory;
    });
    toast.info("Project removed from history");
  };

  const pushHistory = useCallback(() => {
    setHistory(prev => [...prev.slice(-19), JSON.parse(JSON.stringify(segments))]);
  }, [segments]);

  const undo = () => {
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    setSegments(prev);
    setHistory(prevStack => prevStack.slice(0, -1));
    toast.info("Undo: Reverted to previous state");
  };

  const updatePhase = (idx: number, newPhase: string) => {
    const newSegments = [...segments];
    newSegments[idx].phase = newPhase;
    setSegments(newSegments);
  };

  const deleteSegment = (idx: number) => {
    pushHistory();
    const newSegments = segments.filter((_, i) => i !== idx);
    setSegments(newSegments);
    setEditingIdx(null);
    toast.error("Segment deleted");
  };

  const splitSegment = (idx: number) => {
    const newSegments = [...segments];
    const seg = newSegments[idx];
    const startS = timeToSeconds(seg.start_time);
    const endS = timeToSeconds(seg.end_time);

    // Safety: minimum 2 seconds to split
    if (endS - startS < 2) {
      toast.error("Segment too short to split (min 2s)");
      return;
    }

    const midS = Math.floor((startS + endS) / 2);

    const firstSeg = {
      ...seg,
      end_time: secondsToTime(midS),
      duration: midS - startS
    };

    const secondSeg = {
      ...seg,
      start_time: secondsToTime(midS),
      duration: endS - midS
    };

    newSegments.splice(idx, 1, firstSeg, secondSeg);
    setSegments(newSegments);
    toast.success("Segment split into two");
  };

  const mergeWithNext = (idx: number) => {
    if (idx >= segments.length - 1) return;
    const curr = segments[idx];
    const next = segments[idx + 1];
    if (curr.video_id !== next.video_id) {
      toast.warning("Cannot merge segments from different videos");
      return;
    }
    pushHistory();
    const merged: Segment = {
      ...curr,
      end_time: next.end_time,
      duration: timeToSeconds(next.end_time) - timeToSeconds(curr.start_time),
      orig_start_time: curr.orig_start_time,
      orig_end_time: next.orig_end_time
    };
    const newSegments = [...segments];
    newSegments.splice(idx, 2, merged);
    setSegments(newSegments);
    setEditingIdx(null);
    toast.success("Segments merged");
  };

  const insertAfter = (idx: number) => {
    pushHistory();
    const target = segments[idx];
    const newStart = timeToSeconds(target.end_time);
    const newEnd = newStart + 10; // Default 10s
    const newSeg: Segment = {
      ...target,
      phase: "New Phase",
      start_time: secondsToTime(newStart),
      end_time: secondsToTime(newEnd),
      orig_start_time: secondsToTime(newStart),
      orig_end_time: secondsToTime(newEnd),
      duration: 10
    };
    const newSegments = [...segments];
    newSegments.splice(idx + 1, 0, newSeg);
    setSegments(newSegments);
    setEditingIdx(idx + 1);
    toast.success("New segment inserted");
  };
  const [collapsedVideos, setCollapsedVideos] = useState<Record<string, boolean>>({});

  // File Browser State
  const [showBrowser, setShowBrowser] = useState(false);
  const [browserMode, setBrowserMode] = useState<"csv" | "folder">("folder");
  const [fsItems, setFsItems] = useState<FSItem[]>([]);
  const [currentFsPath, setCurrentFsPath] = useState("/");
  const [browserLoading, setBrowserLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);

  // Derived Values
  const phases = useMemo(() => Array.from(new Set(segments.map(s => s.phase))), [segments]);

  const groupedSegments = useMemo(() => {
    const groups: Record<string, Segment[]> = {};
    segments.forEach((seg) => {
      const key = seg.video_id || "Unrecognized Video";
      if (!groups[key]) groups[key] = [];
      groups[key].push(seg);
    });
    return groups;
  }, [segments]);

  const filteredGroups = useMemo(() => {
    const result: Record<string, Segment[]> = {};
    Object.entries(groupedSegments).forEach(([vidId, segs]) => {
      const filtered = segs.filter(s => {
        const isModified = s.start_time !== s.orig_start_time || s.end_time !== s.orig_end_time;
        const matchModified = !showEditedOnly || isModified;

        const matchSearch = s.phase.toLowerCase().includes(searchTerm.toLowerCase()) ||
          s.start_time.includes(searchTerm) ||
          s.end_time.includes(searchTerm);
        const matchPhase = !selectedPhaseFilter || s.phase === selectedPhaseFilter;

        if (searchTerm.match(/^\d{2}:\d{2}:\d{2}$/)) {
          const searchSec = timeToSeconds(searchTerm);
          const startSec = timeToSeconds(s.start_time);
          const endSec = timeToSeconds(s.end_time);
          return searchSec >= startSec && searchSec <= endSec && matchPhase && matchModified;
        }
        return matchSearch && matchPhase && matchModified;
      });
      if (filtered.length > 0) result[vidId] = filtered;
    });
    return result;
  }, [groupedSegments, searchTerm, selectedPhaseFilter, showEditedOnly]);

  // File Browser Actions
  const loadFsDir = useCallback(async (path: string) => {
    setBrowserLoading(true);
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/fs/ls?path=${encodeURIComponent(path)}`);
      setFsItems(resp.data.items);
      setCurrentFsPath(resp.data.current_path);
    } catch (e) {
      toast.error("Failed to read directory");
    } finally {
      setBrowserLoading(false);
    }
  }, []);

  const openBrowser = (mode: "csv" | "folder") => {
    setBrowserMode(mode);
    setShowBrowser(true);
    loadFsDir(currentFsPath || "/");
  };

  const selectFsItem = (item: FSItem) => {
    if (item.is_dir) {
      loadFsDir(item.path);
    } else {
      if (browserMode === "csv" && item.name.endsWith(".csv")) {
        const nameFromPath = item.name.replace('.csv', '');
        setImportInputs(prev => ({
          ...prev,
          csv: item.path,
          projectName: (prev.projectName === "" || prev.projectName === (prev.csv.split(/[\\/]/).pop()?.replace('.csv', '') || ""))
            ? nameFromPath
            : prev.projectName
        }));
        setShowBrowser(false);
      }
    }
  };

  const confirmFolder = () => {
    if (browserMode === "folder") {
      setImportInputs(prev => ({ ...prev, folder: currentFsPath }));
      setShowBrowser(false);
    }
  };

  const handleLoadProject = async () => {
    if (!importInputs.csv || !importInputs.folder) {
      toast.error("Please provide both CSV path and Video folder");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/project/load`, {
        csv_path: importInputs.csv,
        folder_path: importInputs.folder
      });

      // PROJECT ISOLATION: Reset everything
      setSegments([]);
      setHistory([]);
      setCurrentSegment(null);
      setEditingIdx(null);

      const loadedSegments = response.data.segments.map((s: any) => ({
        ...s,
        orig_start_time: s.start_time,
        orig_end_time: s.end_time
      }));

      // SYNC VALIDATION: Check for missing videos
      const videoMappings = (response.data.video_mappings || {}) as Record<string, any>;
      const missingVideos = Array.from(new Set(loadedSegments.map((s: any) => s.video_id)))
        .filter((vidId: any) => !videoMappings[vidId]);

      if (missingVideos.length > 0) {
        toast.warning(`Synchronization Alert: Missing ${missingVideos.length} video files in folder.`, {
          description: `Missing: ${missingVideos.join(', ')}`,
          duration: 5000
        });
      } else {
        toast.success("Project synchronized successfully");
      }

      setSegments(loadedSegments);
      setCsvPath(importInputs.csv);
      setFolderPath(importInputs.folder);

      const defaultName = importInputs.csv.split('/').pop()?.replace('.csv', '') || "Untitled Project";
      const finalName = importInputs.projectName || defaultName;
      setProjectName(finalName);

      saveToHistory(importInputs.csv, importInputs.folder, finalName);
      setShowImport(false);
    } catch (error: any) {
      toast.error("Load Failed: " + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (segments.length === 0) return;
    try {
      const resp = await axios.post(`${API_BASE_URL}/api/export`, { segments });
      const blob = new Blob([resp.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');

      // DYNAMIC NAMING: original_modified.csv
      const originalFilename = csvPath.split('/').pop()?.replace('.csv', '') || 'segments';
      a.href = url;
      a.download = `${originalFilename}_modified.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      toast.success("Modified CSV Exported Successfully");
    } catch (e) {
      toast.error("Export Failed");
    }
  };

  const handleSelectSegment = (seg: Segment) => {
    const idx = segments.indexOf(seg);
    // Transactional Rollback: if editing another segment, restore backup
    if (editingIdx !== null && editingIdx !== idx && segmentsBackup) {
      setSegments(segmentsBackup);
      setSegmentsBackup(null);
      setEditingIdx(null);
      toast.info("Unsaved changes discarded");
    }

    setCurrentSegment(seg);
    if (videoRef.current) {
      if (seg.video_path === currentSegment?.video_path) {
        videoRef.current.currentTime = timeToSeconds(seg.start_time);
      }
    }
  };

  const startEditing = (idx: number, seg: Segment) => {
    // Transactional Rollback if jumping while editing
    if (editingIdx !== null && editingIdx !== idx && segmentsBackup) {
      setSegments(segmentsBackup);
    }

    const s = timeToSeconds(seg.start_time);
    const e = timeToSeconds(seg.end_time);

    setSegmentsBackup(JSON.parse(JSON.stringify(segments)));
    setEditValues({
      start: s,
      end: e,
      duration: e - s,
      phase: seg.phase
    });
    setEditingIdx(idx);
    setCurrentSegment(seg);
  };

  const updateSegmentValues = (idx: number, start: number, end: number) => {
    const newSegments = [...segments];
    const targetVidId = newSegments[idx].video_id;
    newSegments[idx].start_time = secondsToTime(start);
    newSegments[idx].end_time = secondsToTime(end);
    newSegments[idx].duration = end - start;

    // Elastic Push Logic: Recalculate all subsequent segments for this video
    for (let i = idx + 1; i < newSegments.length; i++) {
      if (newSegments[i].video_id !== targetVidId) continue;

      const prevIdx = newSegments.slice(0, i).findLastIndex(s => s.video_id === targetVidId);
      if (prevIdx === -1) continue;

      const prevEnd = timeToSeconds(newSegments[prevIdx].end_time);
      const origStart = timeToSeconds(newSegments[i].orig_start_time);

      // Elastic: Try to snap back to orig_start_time if possible, otherwise push to prevEnd
      const newStart = Math.max(prevEnd, origStart);
      const duration = newSegments[i].duration;

      newSegments[i].start_time = secondsToTime(newStart);
      newSegments[i].end_time = secondsToTime(newStart + duration);
    }
    setSegments(newSegments);
  };

  const toggleVideo = (vidId: string) => {
    setCollapsedVideos(prev => ({ ...prev, [vidId]: !prev[vidId] }));
  };

  return (
    <div className="min-h-screen bg-[#060608] text-zinc-100 p-6 selection:bg-blue-500/30">
      <header className="flex items-center justify-between mb-8 border-b border-white/5 pb-6">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-600 rounded-2xl flex items-center justify-center shadow-2xl shadow-blue-500/20 ring-1 ring-white/10">
            <Film className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight flex items-center gap-2 italic">
              {projectName.includes('.csv') ? projectName.replace('.csv', '') : projectName}
            </h1>
            <p className="text-[10px] text-zinc-500 font-mono tracking-widest uppercase flex items-center gap-1.5 opacity-60">
              {folderPath ? <><Check className="w-3 h-3 text-emerald-500" /> SYNCED: {folderPath.split('/').pop()}</> : "Awaiting Project Parameters"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex gap-4">
            {history.length > 0 && (
              <Button
                variant="outline"
                onClick={undo}
                className="bg-zinc-900/50 border-white/10 hover:bg-white hover:text-black rounded-2xl h-12 px-6"
              >
                <Undo2 className="w-4 h-4 mr-2" /> Undo
              </Button>
            )}
            <Dialog open={showImport} onOpenChange={setShowImport}>
              <DialogTrigger asChild>
                <Button variant="outline" className="bg-zinc-900/50 border-white/5 hover:bg-zinc-800 transition-all rounded-xl px-5">
                  <Plus className="w-4 h-4 mr-2" /> Load Project
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-zinc-950 border-white/5 text-white max-w-xl rounded-3xl p-8 shadow-2xl">
                <DialogHeader className="mb-6">
                  <DialogTitle className="text-2xl font-black italic">Project Configuration</DialogTitle>
                  <DialogDescription className="text-zinc-500 italic">Browse your local files to initialize analytics.</DialogDescription>
                </DialogHeader>
                <div className="space-y-6 py-4">
                  <div className="space-y-2">
                    <label className="text-[10px] uppercase text-zinc-500 font-black tracking-widest ml-1">Project Name (Optional)</label>
                    <Input
                      placeholder="My Analytics Project"
                      value={importInputs.projectName}
                      onChange={e => {
                        setImportInputs(prev => ({ ...prev, projectName: e.target.value }));
                      }}
                      className="bg-zinc-900 border-white/5 h-12 rounded-xl focus:ring-blue-500/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] uppercase text-zinc-500 font-black tracking-widest ml-1">Absolute CSV Path</label>
                    <div className="flex gap-2">
                      <Input
                        placeholder="/Users/name/Desktop/results/case_1.csv"
                        value={importInputs.csv}
                        onChange={e => {
                          const val = e.target.value;
                          const name = val.split(/[\\/]/).pop()?.replace('.csv', '') || "";
                          setImportInputs(prev => ({
                            ...prev,
                            csv: val,
                            projectName: (prev.projectName === "" || prev.projectName === (prev.csv.split(/[\\/]/).pop()?.replace('.csv', '') || ""))
                              ? name
                              : prev.projectName
                          }));
                        }}
                        className="bg-zinc-900 border-white/5 h-12 rounded-xl focus:ring-blue-500/20"
                      />
                      <Button onClick={() => openBrowser("csv")} variant="secondary" className="h-12 px-5 rounded-xl bg-zinc-800 hover:bg-zinc-700">
                        <Search className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] uppercase text-zinc-500 font-black tracking-widest ml-1">Video Folder Path</label>
                    <div className="flex gap-2">
                      <Input
                        placeholder="/Users/name/Desktop/data/videos/1/"
                        value={importInputs.folder}
                        onChange={e => setImportInputs(prev => ({ ...prev, folder: e.target.value }))}
                        className="bg-zinc-900 border-white/5 h-12 rounded-xl focus:ring-blue-500/20"
                      />
                      <Button onClick={() => openBrowser("folder")} variant="secondary" className="h-12 px-5 rounded-xl bg-zinc-800 hover:bg-zinc-700">
                        <FolderOpen className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  {projectHistory.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-white/5">
                      <label className="text-[10px] uppercase text-zinc-500 font-black tracking-widest ml-1">Recent Projects</label>
                      <div className="grid grid-cols-1 gap-2 max-h-32 overflow-y-auto pr-2 custom-scrollbar">
                        {projectHistory.map((h, i) => (
                          <div key={i} className="group relative">
                            <button
                              onClick={() => setImportInputs({ csv: h.csv, folder: h.folder, projectName: h.name })}
                              className="w-full text-left p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-all border border-transparent hover:border-white/10 flex flex-col pr-12"
                            >
                              <p className="text-xs font-black text-zinc-200 group-hover:text-white truncate">{h.name}</p>
                              <p className="text-[9px] text-zinc-600 truncate">{h.csv}</p>
                            </button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={(e) => deleteFromHistory(e, h.csv, h.folder)}
                              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 text-zinc-600 hover:text-rose-500 hover:bg-rose-500/10 opacity-0 group-hover:opacity-100 transition-all rounded-lg"
                            >
                              <X className="w-4 h-4" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <DialogFooter className="mt-8">
                  <Button
                    onClick={handleLoadProject}
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-500 text-white h-14 rounded-2xl font-black italic shadow-xl shadow-blue-500/20 text-lg transition-all"
                  >
                    {loading ? "INITIALIZING..." : "START ANALYSIS"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Button
              onClick={handleExport}
              disabled={segments.length === 0}
              variant="outline"
              className="bg-zinc-950 border-white/5 hover:bg-zinc-900 text-zinc-400 hover:text-white rounded-xl px-5 h-10"
            >
              <Download className="w-4 h-4 mr-2" /> Export CSV
            </Button>
          </div>
        </div>
      </header>

      {/* Integrated File Browser Dialog */}
      <Dialog open={showBrowser} onOpenChange={setShowBrowser}>
        <DialogContent className="bg-zinc-950 border-white/5 text-white max-w-2xl rounded-3xl p-6 shadow-3xl h-[80vh] flex flex-col">
          <DialogHeader className="mb-4">
            <DialogTitle className="text-xl font-black italic flex items-center gap-2">
              <Folder className="w-5 h-5 text-blue-500" />
              {browserMode === "csv" ? "Select CSV Result" : "Select Video Directory"}
            </DialogTitle>
            <div className="flex items-center gap-2 bg-white/5 p-2 rounded-xl mt-4">
              <Button variant="ghost" size="icon" onClick={() => loadFsDir(currentFsPath.split('/').slice(0, -1).join('/') || '/')} className="w-8 h-8 rounded-lg">
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <span className="text-[10px] font-mono text-zinc-500 truncate flex-1">{currentFsPath}</span>
              <Button variant="ghost" size="icon" onClick={() => loadFsDir("/")} className="w-8 h-8 rounded-lg">
                <Home className="w-4 h-4" />
              </Button>
            </div>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto px-2 custom-scrollbar space-y-1">
            {browserLoading ? (
              Array.from({ length: 15 }).map((_, i) => <Skeleton key={i} className="h-10 w-full bg-white/5 rounded-lg" />)
            ) : (
              fsItems.map((item, idx) => (
                <div
                  key={idx}
                  onClick={() => selectFsItem(item)}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-white/5 cursor-pointer group transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {item.is_dir ? <Folder className="w-4 h-4 text-amber-500" /> : <FileIcon className="w-4 h-4 text-blue-400" />}
                    <span className={`text-xs font-medium ${item.name === ".." ? "text-blue-500" : "text-zinc-300 group-hover:text-white"}`}>
                      {item.name}
                    </span>
                  </div>
                  {!item.is_dir && <span className="text-[10px] text-zinc-600 font-mono">{(item.size || 0) / 1024 > 1024 ? `${((item.size || 0) / 1048576).toFixed(1)}MB` : `${((item.size || 0) / 1024).toFixed(0)}KB`}</span>}
                </div>
              ))
            )}
          </div>

          <DialogFooter className="mt-4 pt-4 border-t border-white/5">
            {browserMode === "folder" && (
              <Button onClick={confirmFolder} className="w-full bg-blue-600 hover:bg-blue-500 rounded-xl h-12 font-black italic">
                SELECT THIS FOLDER
              </Button>
            )}
            <Button variant="ghost" onClick={() => setShowBrowser(false)} className="text-zinc-500 hover:text-white">Cancel</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-8 space-y-6">
          <Card className="bg-zinc-900/10 border-white/5 overflow-hidden backdrop-blur-3xl ring-1 ring-white/5 border-none shadow-3xl rounded-[3rem]">
            <div className="flex justify-center bg-black p-8">
              <div style={{ width: '80%' }} className="aspect-video bg-zinc-950 relative group rounded-[2rem] overflow-hidden border border-white/5 shadow-inner">
                {currentSegment?.video_path ? (
                  <video
                    key={currentSegment.video_path}
                    ref={videoRef}
                    src={`${API_BASE_URL}/api/video?path=${encodeURIComponent(currentSegment.video_path)}`}
                    controls
                    className="w-full h-full"
                    autoPlay
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center text-zinc-800">
                    <Film className="w-20 h-20 mb-6 opacity-5 animate-pulse" />
                    <p className="text-xs uppercase font-black tracking-[0.3em] opacity-40 italic">Awaiting Visual Stream</p>
                  </div>
                )}
              </div>
            </div>

            <CardContent className="p-10">
              <div className="flex items-center justify-between mb-10">
                <div className="space-y-2">
                  <Badge variant="outline" className={`py-1 px-4 border-dashed rounded-full text-[10px] font-black tracking-widest ${getPhaseColor(currentSegment?.phase || "")}`}>
                    {currentSegment?.phase || "STANDBY"}
                  </Badge>
                  <h2 className="text-4xl font-black tracking-tighter text-white italic">SEGMENT ANALYTICS</h2>
                  <p className="text-[10px] text-zinc-600 font-mono tracking-tighter uppercase">Source: {currentSegment?.video_id || "None"}</p>
                </div>
                <div className="bg-white/5 px-6 py-4 rounded-3xl border border-white/5 shadow-inner">
                  <p className="text-[10px] uppercase text-zinc-500 font-black tracking-widest mb-1 text-center">Duration</p>
                  <p className="text-3xl font-mono font-black text-blue-400">{currentSegment?.duration.toFixed(1)}<span className="text-xs text-zinc-600 ml-1 italic">sec</span></p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6">
                <div className="p-8 bg-zinc-900/40 rounded-[2.5rem] border border-white/5 space-y-8 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 blur-3xl rounded-full -translate-y-1/2 translate-x-1/2" />

                  <div className="flex items-center gap-2 text-zinc-500 relative z-10">
                    <Clock className="w-4 h-4" />
                    <span className="text-[10px] font-black uppercase tracking-[0.2em] italic">Timeline Micro-Calibration</span>
                  </div>

                  {editingIdx !== null ? (
                    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-2 duration-500 relative z-10">
                      <div className="space-y-6">
                        {/* Phase Name Editing with Presets */}
                        <div className="space-y-4 mb-4">
                          <div className="space-y-1 w-full">
                            <p className="text-[10px] text-zinc-600 font-black uppercase tracking-widest italic">Phase Name</p>
                            <input
                              type="text"
                              value={editValues.phase || segments[editingIdx!].phase}
                              onChange={(e) => {
                                setEditValues({ ...editValues, phase: e.target.value });
                                updatePhase(editingIdx!, e.target.value);
                              }}
                              className="text-3xl font-black text-white bg-transparent border-none outline-none italic w-full focus:ring-0 placeholder:opacity-20"
                              placeholder="ENTER PHASE..."
                            />
                          </div>

                          <div className="flex flex-wrap gap-2 pt-1">
                            {PRESET_PHASES.map(preset => (
                              <button
                                key={preset}
                                onClick={() => {
                                  setEditValues({ ...editValues, phase: preset });
                                  updatePhase(editingIdx!, preset);
                                }}
                                className={`px-3 py-1.5 rounded-full text-[9px] font-black uppercase tracking-wider transition-all border ${(editValues.phase || segments[editingIdx!].phase) === preset
                                  ? "bg-white text-black border-white"
                                  : "bg-white/5 text-zinc-500 border-white/5 hover:border-white/20 hover:text-zinc-300"
                                  }`}
                              >
                                {preset}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="h-px bg-white/5 w-full" />

                        <div className="flex justify-between items-end px-2">
                          <div className="space-y-1">
                            <p className="text-[10px] text-zinc-600 font-black uppercase tracking-widest italic">Segment Start</p>
                            <p className="text-3xl font-mono font-black text-white italic">{secondsToTime(editValues.start)}</p>
                          </div>
                          <ArrowRight className="w-8 h-8 text-zinc-800 mb-1" />
                          <div className="text-right space-y-1">
                            <p className="text-[10px] text-zinc-600 font-black uppercase tracking-widest italic">Segment End</p>
                            <p className="text-3xl font-mono font-black text-white italic">{secondsToTime(editValues.end)}</p>
                          </div>
                        </div>
                        <Slider
                          value={[editValues.start, editValues.end]}
                          max={3600}
                          step={1}
                          className="py-6"
                          onValueChange={(vals) => {
                            setEditValues({ ...editValues, start: vals[0], end: vals[1], duration: vals[1] - vals[0] });
                            updateSegmentValues(editingIdx!, vals[0], vals[1]);
                          }}
                        />
                      </div>

                      <div className="grid grid-cols-3 gap-6">
                        <div className="space-y-3">
                          <label className="text-[10px] uppercase text-zinc-500 font-black tracking-widest ml-1 italic">Start Time</label>
                          <Input
                            value={secondsToTime(editValues.start)}
                            onChange={e => {
                              const s = timeToSeconds(e.target.value);
                              setEditValues({ ...editValues, start: s, duration: editValues.end - s });
                              updateSegmentValues(editingIdx!, s, editValues.end);
                            }}
                            className="bg-black/60 border-white/5 rounded-2xl h-14 font-mono text-center text-blue-400 focus:ring-blue-500/10 text-xl"
                          />
                        </div>
                        <div className="space-y-3">
                          <label className="text-[10px] uppercase text-zinc-500 font-black tracking-widest ml-1 italic">Duration (s)</label>
                          <Input
                            type="number"
                            value={editValues.duration}
                            onChange={e => {
                              const d = parseInt(e.target.value) || 0;
                              const newEnd = editValues.start + d;
                              setEditValues({ ...editValues, duration: d, end: newEnd });
                              updateSegmentValues(editingIdx!, editValues.start, newEnd);
                            }}
                            className="bg-black/60 border-white/5 rounded-2xl h-14 font-mono text-center text-amber-500 focus:ring-amber-500/10 text-xl"
                          />
                        </div>
                        <div className="space-y-3">
                          <label className="text-[10px] uppercase text-zinc-500 font-black tracking-widest ml-1 italic">End Time</label>
                          <Input
                            value={secondsToTime(editValues.end)}
                            onChange={e => {
                              const ev = timeToSeconds(e.target.value);
                              setEditValues({ ...editValues, end: ev, duration: ev - editValues.start });
                              updateSegmentValues(editingIdx!, editValues.start, ev);
                            }}
                            className="bg-black/60 border-white/5 rounded-2xl h-14 font-mono text-center text-purple-400 focus:ring-purple-500/10 text-xl"
                          />
                        </div>
                      </div>

                      <div className="h-px bg-white/5 w-full opacity-50" />

                      {/* Operational Toolbar */}
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <Button
                            onClick={() => insertAfter(editingIdx!)}
                            variant="secondary"
                            className="bg-white/5 hover:bg-emerald-500/20 border-white/10 h-16 rounded-2xl flex flex-col items-center justify-center gap-1 group transition-all"
                          >
                            <Plus className="w-5 h-5 text-emerald-400 group-hover:scale-110 transition-transform" />
                            <span className="text-[10px] font-black uppercase tracking-tighter">Insert New After</span>
                          </Button>
                          <Button
                            onClick={() => mergeWithNext(editingIdx!)}
                            disabled={editingIdx === segments.length - 1}
                            variant="secondary"
                            className="bg-white/5 hover:bg-blue-500/20 border-white/10 h-16 rounded-2xl flex flex-col items-center justify-center gap-1 group transition-all"
                          >
                            <Combine className="w-5 h-5 text-blue-400 group-hover:scale-110 transition-transform" />
                            <span className="text-[10px] font-black uppercase tracking-tighter">Merge with Next</span>
                          </Button>
                          <Button
                            onClick={() => splitSegment(editingIdx!)}
                            variant="secondary"
                            className="bg-white/5 hover:bg-orange-500/20 border-white/10 h-16 rounded-2xl flex flex-col items-center justify-center gap-1 group transition-all"
                          >
                            <Scissors className="w-5 h-5 text-orange-400 group-hover:scale-110 transition-transform" />
                            <span className="text-[10px] font-black uppercase tracking-tighter">Split Segment</span>
                          </Button>
                        </div>
                        <Button
                          onClick={() => deleteSegment(editingIdx!)}
                          variant="destructive"
                          className="w-full bg-rose-500/10 hover:bg-rose-500 text-rose-500 hover:text-white border-rose-500/20 h-16 rounded-2xl flex items-center justify-center gap-4 group transition-all"
                        >
                          <Trash2 className="w-6 h-6 group-hover:scale-110 transition-transform" />
                          <span className="text-sm font-black uppercase tracking-widest">Delete This Segment</span>
                        </Button>
                      </div>

                      <Button
                        onClick={() => {
                          pushHistory();
                          setSegmentsBackup(null);
                          setEditingIdx(null);
                          toast.success("Changes committed");
                        }}
                        className="w-full bg-blue-600 hover:bg-blue-500 text-white h-14 rounded-2xl font-black italic shadow-2xl shadow-blue-500/20 text-lg tracking-tight"
                      >
                        VALIDATE REVISION
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => {
                          if (segmentsBackup) {
                            setSegments(segmentsBackup);
                            setSegmentsBackup(null);
                          }
                          setEditingIdx(null);
                          toast.info("Changes discarded");
                        }}
                        className="w-full text-zinc-500 hover:text-white h-10 font-bold uppercase text-[10px] tracking-widest mt-2"
                      >
                        Discard Changes
                      </Button>
                    </div>
                  ) : (
                    <div className="text-center py-10 border-2 border-dashed border-white/5 rounded-[2rem] group hover:border-white/10 transition-all opacity-60">
                      <Edit2 className="w-10 h-10 text-zinc-800 mx-auto mb-4 group-hover:text-zinc-700 group-hover:rotate-12 transition-all" />
                      <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest italic">Target a segment to activate editing engine</p>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-4 space-y-6 max-h-[calc(100vh-140px)] flex flex-col">
          <div className="space-y-5 bg-zinc-900/20 p-8 rounded-[3rem] border border-white/5 backdrop-blur-3xl shrink-0 shadow-3xl">
            <div className="relative group">
              <Search className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-600 group-focus-within:text-blue-500 transition-colors" />
              <input
                placeholder="Find phase or 'HH:MM:SS'..."
                className="w-full bg-black/40 border border-white/5 rounded-2xl py-4 pl-14 pr-4 text-xs font-bold italic focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all placeholder:text-zinc-700"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
              />
            </div>

            <div className="flex flex-wrap gap-1.5">
              <Button
                onClick={() => { setSelectedPhaseFilter(null); setShowEditedOnly(false); }}
                className={`h-7 px-3 text-[9px] font-black uppercase rounded-full transition-all ${selectedPhaseFilter === null && !showEditedOnly ? "bg-white text-black shadow-lg" : "bg-white/5 text-zinc-600 hover:text-zinc-300 border border-white/5"}`}
              >
                All States
              </Button>
              <Button
                onClick={() => setShowEditedOnly(!showEditedOnly)}
                className={`h-7 px-3 text-[9px] font-black uppercase rounded-full border transition-all ${showEditedOnly ? "bg-amber-600 text-white border-amber-500 shadow-lg shadow-amber-500/10" : "bg-white/5 border-white/5 text-zinc-600 hover:text-zinc-300"}`}
              >
                <History className="w-3 h-3 mr-1" /> Edited
              </Button>
              <div className="w-px h-4 bg-white/10 mx-1 self-center" />
              {phases.map(p => (
                <Button
                  key={p}
                  onClick={() => setSelectedPhaseFilter(p)}
                  className={`h-7 px-3 text-[9px] font-black uppercase rounded-full border transition-all ${selectedPhaseFilter === p ? "bg-blue-600 text-white border-blue-500 shadow-lg shadow-blue-500/10" : "bg-white/5 border-white/5 text-zinc-600 hover:text-zinc-300"}`}
                >
                  {p}
                </Button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto pr-3 custom-scrollbar space-y-6">
            <AnimatePresence mode="popLayout">
              {loading ? (
                Array.from({ length: 12 }).map((_, i) => (
                  <Skeleton key={i} className="h-28 w-full bg-zinc-900/10 rounded-[2.5rem]" />
                ))
              ) : Object.keys(filteredGroups).length > 0 ? (
                Object.entries(filteredGroups).map(([vidId, segs]) => {
                  const isCollapsed = collapsedVideos[vidId];
                  return (
                    <div key={vidId} className="space-y-3">
                      <div className="flex items-center justify-between px-2">
                        <div onClick={() => toggleVideo(vidId)} className="flex items-center gap-4 cursor-pointer group">
                          <div className="w-10 h-10 rounded-2xl bg-white/5 flex items-center justify-center group-hover:bg-white/10 transition-all border border-white/5">
                            {isCollapsed ? <ChevronRight className="w-5 h-5 text-zinc-500" /> : <ChevronDown className="w-5 h-5 text-white" />}
                          </div>
                          <p className="text-sm font-black tracking-[0.2em] uppercase text-zinc-400 group-hover:text-white transition-colors">
                            {vidId}
                          </p>
                        </div>
                        <Badge variant="outline" className="text-xs font-mono text-zinc-600 border-white/5 px-3 h-6">{segs.length} PHASES</Badge>
                      </div>

                      {!isCollapsed && segs.map((seg, sIdx) => {
                        const globalIdx = segments.indexOf(seg);
                        const isActive = currentSegment === seg;
                        const phaseColor = getPhaseColor(seg.phase);
                        return (
                          <motion.div
                            key={`${vidId}-${sIdx}`}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            layout
                            onClick={() => handleSelectSegment(seg)}
                            className={`group p-6 rounded-[2.5rem] border transition-all cursor-pointer relative overflow-hidden ${isActive
                              ? "ring-2 ring-white/20 shadow-2xl scale-[1.02] z-20"
                              : "hover:scale-[1.01] hover:bg-white/5 active:scale-100"
                              } ${getPhaseColor(seg.phase).split(' ')[0]} ${isActive ? "border-white/20" : "border-white/5"}`}
                          >
                            <div className="relative z-10">
                              <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-4">
                                  <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-700 ${isActive ? "bg-white text-black shadow-lg" : "bg-black/40 text-zinc-400 group-hover:text-white"}`}>
                                    <Play className={`w-6 h-6 ${isActive ? "fill-black" : ""}`} />
                                  </div>
                                  <div>
                                    <div className="flex items-center gap-3">
                                      <p className={`font-black text-lg tracking-tight italic transition-colors ${isActive ? "text-white" : "text-zinc-100 group-hover:text-white"}`}>{seg.phase}</p>
                                      {(seg.start_time !== seg.orig_start_time || seg.end_time !== seg.orig_end_time) && (
                                        <div className="flex items-center gap-1 text-amber-400">
                                          <History className="w-4 h-4 opacity-80" />
                                          <span className="text-[9px] font-black uppercase tracking-tighter opacity-60">Edited</span>
                                        </div>
                                      )}
                                    </div>
                                    <div className="flex items-center gap-2 mt-2">
                                      <Badge variant="outline" className="text-[9px] px-2.5 py-0 h-5 border-white/20 font-black tracking-widest text-white bg-black/40">DETERMINED</Badge>
                                      {isActive && <Badge className="bg-white text-black border-none text-[9px] font-black h-5 px-2.5 italic animate-pulse">LIVE</Badge>}
                                    </div>
                                  </div>
                                </div>
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  className={`w-10 h-10 rounded-2xl transition-all duration-300 ${isActive ? "bg-white/20 text-white hover:bg-white hover:text-black" : "opacity-0 group-hover:opacity-100 bg-white/5 hover:bg-white hover:text-black text-white"}`}
                                  onClick={(e) => { e.stopPropagation(); startEditing(globalIdx, seg); }}
                                >
                                  <Edit2 className="w-4 h-4" />
                                </Button>
                              </div>
                              <div className="flex items-center justify-between mt-2 pl-1">
                                <div className={`flex flex-col gap-1 ${isActive ? "text-white" : "text-zinc-200 group-hover:text-white"}`}>
                                  {(seg.start_time !== seg.orig_start_time || seg.end_time !== seg.orig_end_time) && (
                                    <div className="flex items-center gap-2 text-[10px] font-mono opacity-40 line-through mb-0.5">
                                      <span>{seg.orig_start_time}</span>
                                      <ArrowRight className="w-2.5 h-2.5 mx-1" />
                                      <span>{seg.orig_end_time}</span>
                                    </div>
                                  )}
                                  <div className="flex items-center gap-4 text-base font-mono font-black italic">
                                    <span className="flex items-center gap-2.5"><Clock className="w-4 h-4" /> {seg.start_time}</span>
                                    <ArrowRight className="w-4 h-4 opacity-20" />
                                    <span>{seg.end_time}</span>
                                  </div>
                                </div>
                                <p className={`text-xs font-black tracking-tighter italic ${isActive ? "text-white/60" : "text-zinc-500 opacity-60"}`}>{seg.duration.toFixed(0)} S</p>
                              </div>
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  );
                })
              ) : (
                <div className="text-center py-20 border-2 border-dashed border-white/5 rounded-[3rem] opacity-20">
                  <Upload className="w-16 h-16 mx-auto mb-4" />
                  <p className="text-xs font-black tracking-widest uppercase">No Active Trajectory</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
