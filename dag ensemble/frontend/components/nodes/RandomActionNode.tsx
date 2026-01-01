
import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

const RandomActionNode = ({ data }: NodeProps) => {

    const result = data.latestData; // This will come from backend execution

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-64 transition-all duration-200 ${data.isExecuting ? 'border-purple-500 ring-2 ring-purple-200' : 'border-purple-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-purple-500 !w-3 !h-3" /> {/* Input now visible for triggering */}

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-purple-100 flex items-center justify-center text-purple-600 font-bold text-xs">RA</div>
                    <div className="font-bold text-sm text-zinc-900">Random Action</div>
                </div>
                {/* Execution Badge */}
                {data.isExecuting && (
                    <div className="w-2 h-2 rounded-full bg-purple-500 animate-ping" />
                )}
            </div>

            <div className="bg-zinc-50 border border-zinc-100 rounded p-2 text-xs">
                <div className="text-[10px] text-zinc-400 uppercase font-semibold mb-1">Generated Output</div>
                {result ? (
                    <div className="font-mono text-zinc-700">
                        <div className={`font-bold ${result.action === 'buy' ? 'text-green-600' : result.action === 'sell' ? 'text-red-600' : 'text-zinc-500'}`}>
                            ACTION: {result.action ? result.action.toUpperCase() : 'N/A'}
                        </div>
                        <div>VOL: {result.volume}</div>
                    </div>
                ) : (
                    <div className="text-zinc-400 italic">No output yet...</div>
                )}
            </div>

            <Handle type="source" position={Position.Right} className="!bg-purple-500 !w-3 !h-3" />
        </div>
    );
};

export default memo(RandomActionNode);
