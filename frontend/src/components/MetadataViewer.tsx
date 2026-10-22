import React, { useState } from 'react';

export const MetadataViewer: React.FC<{ data: Record<string, unknown> }> = ({ data }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-slate-200 rounded-lg p-3 text-xs bg-slate-50">
      <button onClick={() => setOpen(!open)} className="font-medium text-slate-700">
        {open ? 'Hide Forensic Details' : 'Show Forensic Details'}
      </button>
      {open && (
        <pre className="mt-2 text-slate-600 overflow-x-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
};
