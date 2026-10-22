import React from 'react';

export const ModalityBadge: React.FC<{ modality: 'image' | 'video' | 'audio' }> = ({ modality }) => {
  const colors = {
    image: 'bg-blue-100 text-blue-800',
    video: 'bg-purple-100 text-purple-800',
    audio: 'bg-amber-100 text-amber-800',
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${colors[modality]}`}>
      {modality.toUpperCase()}
    </span>
  );
};
