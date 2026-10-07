import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  LayoutDashboard, 
  Image as ImageIcon, 
  Video, 
  Mic, 
  Layers, 
  BarChart3, 
  Eye, 
  TrendingUp, 
  FlaskConical,
  Menu,
  X
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Overview', href: '/', icon: LayoutDashboard },
  { name: 'Image', href: '/image', icon: ImageIcon },
  { name: 'Video', href: '/video', icon: Video },
  { name: 'Audio', href: '/audio', icon: Mic },
  { name: 'Fusion', href: '/multimodal', icon: Layers },
  { name: 'Confidence', href: '/confidence', icon: BarChart3 },
  { name: 'Explainability', href: '/explainability', icon: Eye },
  { name: 'Generalization', href: '/generalization', icon: TrendingUp },
  { name: 'Experiments', href: '/experiments', icon: FlaskConical },
]

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen bg-transparent relative flex flex-col md:flex-row">
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
          />
        )}
      </AnimatePresence>

      <div
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-72 bg-gray-900/60 backdrop-blur-2xl border-r border-white/10 transform transition-transform duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] lg:translate-x-0 lg:static flex-shrink-0 shadow-2xl overflow-y-auto",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex items-center justify-between h-20 px-6 border-b border-white/10 sticky top-0 bg-gray-900/40 backdrop-blur-xl z-10">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-[0_0_20px_rgba(14,165,233,0.5)] transition-all duration-300">
              <span className="text-white font-black text-lg tracking-wider">A</span>
            </div>
            <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 tracking-tight">AEGIS</span>
          </Link>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-2 rounded-xl bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="p-4 space-y-1.5 mt-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                onClick={() => setSidebarOpen(false)}
                className={cn(
                  "flex items-center gap-4 px-4 py-3.5 rounded-xl text-sm font-semibold transition-all duration-300 relative group overflow-hidden",
                  isActive
                    ? "text-white"
                    : "text-gray-400 hover:text-gray-200"
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeNavTab"
                    className="absolute inset-0 bg-gradient-to-r from-primary-500/20 to-indigo-500/10 border border-primary-500/30 rounded-xl -z-10"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                <div className={cn(
                  "p-2 rounded-lg transition-colors duration-300",
                  isActive ? "bg-primary-500/20 text-primary-400" : "bg-gray-800/50 group-hover:bg-gray-700/50"
                )}>
                  <item.icon className="w-5 h-5" />
                </div>
                {item.name}
              </Link>
            )
          })}
        </nav>
      </div>

      <div className="flex-1 flex flex-col min-h-screen relative w-full lg:w-auto">
        <div className="sticky top-0 z-30 bg-gray-900/40 backdrop-blur-xl border-b border-white/5 h-20 transition-all duration-300">
          <div className="flex items-center justify-between h-full px-6">
             <div className="flex items-center gap-4">
               <button
                 onClick={() => setSidebarOpen(true)}
                 className="lg:hidden p-2 rounded-xl bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
               >
                 <Menu className="w-6 h-6" />
               </button>
               <h1 className="text-lg font-semibold text-gray-200">
                  {navigation.find(n => n.href === location.pathname)?.name || "Dashboard"}
               </h1>
             </div>
             <div className="flex items-center gap-4">
                <div className="hidden sm:flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">System Online</span>
                </div>
             </div>
          </div>
        </div>

        <main className="flex-1 p-6 lg:p-10 max-w-[1600px] mx-auto w-full">
          <AnimatePresence mode="wait">
             <motion.div
               key={location.pathname}
               initial={{ opacity: 0, y: 15 }}
               animate={{ opacity: 1, y: 0 }}
               exit={{ opacity: 0, y: -15 }}
               transition={{ duration: 0.3, ease: 'easeOut' }}
             >
               {children}
             </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}
