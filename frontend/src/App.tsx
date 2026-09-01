import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import ImageDetection from './pages/ImageDetection'
import VideoDetection from './pages/VideoDetection'
import AudioDetection from './pages/AudioDetection'
import MultimodalDetection from './pages/MultimodalDetection'
import Confidence from './pages/Confidence'
import Explainability from './pages/Explainability'
import Generalization from './pages/Generalization'
import Experiments from './pages/Experiments'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/image" element={<ImageDetection />} />
          <Route path="/video" element={<VideoDetection />} />
          <Route path="/audio" element={<AudioDetection />} />
          <Route path="/multimodal" element={<MultimodalDetection />} />
          <Route path="/confidence" element={<Confidence />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/generalization" element={<Generalization />} />
          <Route path="/experiments" element={<Experiments />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
