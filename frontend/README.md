# PhasePro MAX - Frontend Guide

This directory contains the codebase for the PhasePro MAX web interface. This tool provides intuitive, high-level editing capabilities for correcting and optimizing machine-generated surgical video segments.

## Quick Start

### 1. Start the Backend Server (Required)
The frontend relies on the backend API to read files and stream videos.
From the project root:
```bash
pip install fastapi uvicorn pandas
python server.py
```

### 2. Start the Frontend
From the `frontend` directory:
1. 'cd frontend'
2. **Install Dependencies**: `npm install`
3. **Run Dev Server**: `npm run dev`
4. **Open Browser**: Navigate to [http://localhost:3000](http://localhost:3000)

## Key Features

### Project Management
- **Load Project**: Click "Load Project" to initialize.
- **Absolute CSV Path**: Full path to your model's output CSV.
- **Video Folder Path**: Full path to the folder containing case videos.
- **Persistent History**: The system automatically remembers your **Recent Projects** for one-click loading.

### Advanced Timeline Editing
- **Start/End Adjustments**: Edit timestamps directly via text input or slider.
- **Duration-Based Editing**: Modifying "Duration (s)" keeps the Start Time fixed and automatically updates the End Time.
- **Elastic Push Logic**: Smart timeline management. When a segment is moved or resized, subsequent segments shift automatically and "snap back" to their original positions if space becomes available.
- **Multi-Video Sync**: Seamlessly handles cases spanning multiple video files, switching the player automatically based on segment selection.

### Export
- **Export CSV**: Generates a refined CSV.
- **Dynamic Naming**: Files are automatically named as `[original_name]_modified.csv`.

## Tech Stack
- **Framework**: Next.js 15
- **Styling**: Tailwind CSS
- **Components**: Lucide Icons, Shadcn UI, Framer Motion
- **Networking**: Axios
