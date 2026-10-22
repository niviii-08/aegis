import React from 'react';

export const AudioWaveform: React.FC = () => {
  return (
    <svg className="w-full h-8 text-sky-500" viewBox="0 0 100 20" fill="currentColor">
      <rect x="5" y="6" width="3" height="8" rx="1.5" />
      <rect x="15" y="2" width="3" height="16" rx="1.5" />
      <rect x="25" y="5" width="3" height="10" rx="1.5" />
      <rect x="35" y="1" width="3" height="18" rx="1.5" />
      <rect x="45" y="4" width="3" height="12" rx="1.5" />
      <rect x="55" y="8" width="3" height="4" rx="1.5" />
      <rect x="65" y="3" width="3" height="14" rx="1.5" />
      <rect x="75" y="6" width="3" height="8" rx="1.5" />
      <rect x="85" y="2" width="3" height="16" rx="1.5" />
    </svg>
  );
};
