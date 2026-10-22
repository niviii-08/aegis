import React from 'react';

interface Props {
  confidence: number;
  isFake: boolean;
}

export const ConfidenceMeter: React.FC<Props> = ({ confidence, isFake }) => {
  const pct = Math.round(confidence * 100);
  return (
    <div className="flex flex-col gap-1 w-full max-w-xs">
      <div className="flex justify-between text-sm font-semibold">
        <span>Confidence</span>
        <span className={isFake ? 'text-red-500' : 'text-emerald-500'}>{pct}%</span>
      </div>
      <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${isFake ? 'bg-red-500' : 'bg-emerald-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};
