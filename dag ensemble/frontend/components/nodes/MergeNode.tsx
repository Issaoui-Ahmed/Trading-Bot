import React, { memo, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';

const MergeNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const [mergeType, setMergeType] = useState(data.mergeType || 'concat');

    const handleMergeTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newVal = e.target.value;
        setMergeType(newVal);

        setNodes((nds) =>
            nds.map((node) => {
                if (node.id === id) {
                    return { ...node, data: { ...node.data, mergeType: newVal } };
                }
                return node;
            })
        );
    };

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-64 transition-all duration-200 ${data.isExecuting ? 'border-orange-500 ring-2 ring-orange-200 shadow-orange-100' : 'border-orange-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-orange-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-orange-100 flex items-center justify-center text-orange-600 font-bold text-xs">M</div>
                    <div className="font-bold text-sm text-zinc-900">Merge Node</div>
                </div>
                {/* Execution Time Badge */}
                {data.executionTime !== undefined && (
                    <div className={`text-[10px] font-mono px-1.5 py-0.5 rounded border flex items-center gap-1 ${data.isExecuting ? 'bg-orange-100 text-orange-700 border-orange-200 animate-pulse' : 'bg-zinc-50 text-zinc-500 border-zinc-200'
                        }`}>
                        {data.isExecuting && <div className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-ping" />}
                        {data.executionTime.toFixed(2)}ms
                    </div>
                )}
            </div>

            <div className="space-y-3 mb-2">
                <label className="text-[10px] text-zinc-500 font-semibold uppercase block">Merge Strategy</label>
                <select
                    className="w-full text-xs border border-zinc-200 rounded p-1"
                    value={mergeType}
                    onChange={handleMergeTypeChange}
                >
                    <option value="concat">Concat (Axis 1)</option>
                    <option value="inner_join">Inner Join (Index)</option>
                    <option value="outer_join">Outer Join (Index)</option>
                </select>
            </div>
            <p className="text-[10px] text-zinc-400 italic mb-2">
                Connect multiple nodes to merge their outputs.
            </p>

            {data.latestData && (
                <div className="mt-3 border-t border-zinc-100 pt-2">
                    <p className="text-[10px] text-zinc-500 font-semibold mb-1">Merged Output</p>

                    {/* Error Display */}
                    {(data.latestData.error) ? (
                        <div className="bg-red-50 border border-red-200 rounded p-2 text-[10px] text-red-600 font-mono break-words">
                            {data.latestData.error}
                        </div>
                    ) : (data.latestData.status) ? (
                        /* Status Display (Waiting) */
                        <div className="bg-yellow-50 border border-yellow-200 rounded p-2 text-[10px] text-yellow-700 font-mono italic">
                            {data.latestData.status}
                        </div>
                    ) : (
                        /* Success Data Display */
                        <div className="bg-zinc-50 rounded p-2 overflow-x-auto max-h-32 overflow-y-auto">
                            <pre className="text-[9px] text-zinc-600 font-mono">
                                {JSON.stringify(data.latestData, null, 2)}
                            </pre>
                        </div>
                    )}
                </div>
            )}

            <Handle type="source" position={Position.Right} className="!bg-orange-500 !w-3 !h-3" />
        </div>
    );
};

export default memo(MergeNode);
