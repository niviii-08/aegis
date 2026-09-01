# AEGIS Frontend Dashboard

Professional ML research portfolio dashboard for the AEGIS deepfake detection system.

## Features

- **Multi-modal Detection Interface**: Upload and analyze images, videos, and audio files
- **Real-time Inference**: Direct integration with AEGIS FastAPI backend
- **Research Visualization**: Display of calibration, explainability, and generalization analysis
- **Professional UI**: Clean, research-focused interface suitable for ML portfolios
- **Responsive Design**: Works across desktop and tablet devices
- **Status Monitoring**: Real-time system health and model status tracking

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Axios** - HTTP client
- **Lucide React** - Icons
- **Recharts** - Data visualization

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment configuration
cp .env.example .env

# Configure API URL
# Edit .env and set VITE_API_URL to your backend URL
```

## Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Configuration

Environment variables in `.env`:

```bash
VITE_API_URL=http://localhost:8000
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── Layout.tsx          # Main layout with navigation
│   ├── lib/
│   │   ├── api.ts              # API client and types
│   │   └── utils.ts            # Utility functions
│   ├── pages/
│   │   ├── Overview.tsx        # System overview
│   │   ├── ImageDetection.tsx  # Image detection interface
│   │   ├── VideoDetection.tsx  # Video detection interface
│   │   ├── AudioDetection.tsx  # Audio detection interface
│   │   ├── MultimodalDetection.tsx  # Multimodal fusion
│   │   ├── Confidence.tsx      # Calibration analysis
│   │   ├── Explainability.tsx  # Explainability methods
│   │   ├── Generalization.tsx  # Generalization analysis
│   │   └── Experiments.tsx     # Experiment results
│   ├── App.tsx                 # Main app component
│   ├── main.tsx                # Entry point
│   └── index.css               # Global styles
├── public/                     # Static assets
├── index.html                  # HTML template
├── package.json                # Dependencies
├── vite.config.ts             # Vite configuration
├── tailwind.config.js         # Tailwind configuration
└── tsconfig.json              # TypeScript configuration
```

## Pages Overview

### Overview
System health monitoring, model status, and quick stats

### Detection Pages (Image/Video/Audio)
- File upload with drag-and-drop support
- Real-time prediction results
- Confidence metrics with calibration status
- Research context and methodology

### Multimodal Detection
- Multi-file upload interface
- Fusion prediction results
- Cross-modal analysis context

### Confidence & Calibration
- Temperature scaling status per modality
- Reliability metrics (ECE, Brier score)
- Calibration methodology explanation

### Explainability
- Grad-CAM and saliency analysis methods
- Per-modality explanation capabilities
- Integration status

### Generalization Analysis
- Generator-based split methodology
- Generalization gap framework
- Research questions and data requirements

### Experiments
- Baseline model performance
- Ablation study results
- Training configuration details
- Results file locations

## API Integration

The frontend communicates with the AEGIS FastAPI backend:

```typescript
// Example API usage
import { apiClient } from '@/lib/api'

// Health check
const health = await apiClient.getHealth()

// Image prediction
const result = await apiClient.predictImage(file)

// Video prediction
const result = await apiClient.predictVideo(file)

// Audio prediction
const result = await apiClient.predictAudio(file)

// Multimodal prediction
const result = await apiClient.predictMultimodal({
  image: imageFile,
  video: videoFile,
  audio: audioFile
})
```

## Data Display Philosophy

The dashboard follows a research-focused approach:

1. **No Fake Metrics**: All displayed data comes from backend API or result files
2. **TBD Values**: Unavailable data is clearly marked as "TBD" rather than fabricated
3. **Status Indicators**: Clear status badges show data availability
4. **Research Context**: Each section includes methodology and research questions
5. **Professional Presentation**: Clean, academic-style visualization

## Styling

The dashboard uses a custom color scheme optimized for research presentations:

- **Primary Blue**: Professional, trustworthy
- **Research Colors**: Purple, green, red, orange for different research aspects
- **Status Colors**: Green (healthy), yellow (warning), red (error)
- **Gradients**: Subtle gradients for emphasis

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Performance

- Lazy loading of pages
- Optimized bundle size
- Efficient state management
- Responsive image handling

## Development Notes

- The dashboard is designed to work with the AEGIS FastAPI backend
- All prediction endpoints require proper model checkpoints
- Calibration data is loaded from backend calibration files
- Research results sections show framework and expected analyses
- Real data will populate once model training and evaluation complete

## Future Enhancements

- Interactive chart components using Recharts
- Real-time result streaming
- Export functionality for reports
- Comparison views across experiments
- Advanced explainability visualizations
- Custom dashboard configuration

## License

This frontend is part of the AEGIS project. See the main project license for details.
