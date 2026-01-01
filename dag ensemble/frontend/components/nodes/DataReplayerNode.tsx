import React, { memo, useCallback, useEffect, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';

const FREQUENCIES = ['instant', '1s', '5s', '10s', '30s', '1m'];

const DataReplayerNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const [datasets, setDatasets] = useState<string[]>([]);

    // Default configuration
    const frequency = data.frequency || '5s';
    const selectedDataset = data.dataset || '';

    useEffect(() => {
        // Fetch available datasets
        fetch('http://localhost:8001/datasets')
            .then(res => res.json())
            .then(data => setDatasets(data.datasets || []))
            .catch(err => console.error('Failed to load datasets', err));
    }, []);

    const updateData = useCallback((key: string, value: string) => {
        setNodes((nodes) =>
            nodes.map((node) => {
                if (node.id === id) {
                    return { ...node, data: { ...node.data, [key]: value } };
                }
                return node;
            })
        );
    }, [id, setNodes]);

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-72 transition-all duration-200 ${data.isExecuting ? 'border-purple-500 ring-2 ring-purple-200 shadow-purple-100' : 'border-purple-200'
            }`}>
            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-purple-100 flex items-center justify-center text-purple-600 font-bold text-xs">RPL</div>
                    <div className="font-bold text-sm text-zinc-900">Data Replayer</div>
                </div>
                {/* Execution Time Badge */}
                {data.executionTime !== undefined && (
                    <div className={`text-[10px] font-mono px-1.5 py-0.5 rounded border flex items-center gap-1 ${data.isExecuting ? 'bg-purple-100 text-purple-700 border-purple-200 animate-pulse' : 'bg-zinc-50 text-zinc-500 border-zinc-200'
                        }`}>
                        {data.isExecuting && <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-ping" />}
                        {data.executionTime.toFixed(2)}ms
                    </div>
                )}
            </div>

            {/* Replay Progress */}
            {data.replayerStats && (
                <div className="mb-3">
                    <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                        <span className="font-semibold">PROGRESS</span>
                        <div className="flex gap-2">
                            <span className="font-mono text-zinc-400">
                                {data.replayerStats.elapsed !== undefined
                                    ? new Date(data.replayerStats.elapsed * 1000).toISOString().substr(11, 8)
                                    : "00:00:00"}
                            </span>
                            <span className="font-mono text-purple-600 font-bold">{data.replayerStats.progress.toFixed(1)}%</span>
                        </div>
                    </div>

                    {/* [NEW] Simulated Timestamp Display */}
                    {data.replayerStats.current_timestamp && (
                        <div className="flex justify-between text-[10px] mb-1">
                            <span className="text-zinc-500 font-semibold">SIM TIME</span>
                            <span className="font-mono text-zinc-700 font-bold">
                                {new Date(data.replayerStats.current_timestamp * 1000).toLocaleString()}
                            </span>
                        </div>
                    )}

                    <div className="w-full bg-zinc-100 rounded-full h-1.5 overflow-hidden mb-1">
                        <div
                            className="bg-purple-500 h-full transition-all duration-300"
                            style={{ width: `${data.replayerStats.progress}%` }}
                        />
                    </div>
                    <div className="flex justify-between text-[10px] text-zinc-400 font-mono">
                        <span>{data.replayerStats.current}</span>
                        <span>/</span>
                        <span>{data.replayerStats.total}</span>
                    </div>
                </div>
            )}



            <div className="space-y-3 mb-4">
                <div className="text-[10px] text-zinc-500 font-semibold uppercase">Configuration</div>
                <div className="bg-zinc-50 border border-zinc-100 rounded p-2 space-y-2">

                    {/* Dataset Selection */}
                    <div className="flex flex-col gap-1">
                        <label className="text-xs text-zinc-500 font-medium">Dataset</label>
                        <select
                            className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-700 outline-none focus:border-purple-400"
                            value={selectedDataset}
                            onChange={(e) => updateData('dataset', e.target.value)}
                        >
                            <option value="">Select Dataset...</option>
                            {datasets.map(d => (
                                <option key={d} value={d}>{d}</option>
                            ))}
                        </select>
                    </div>

                    {/* Frequency Selection */}
                    <div className="flex flex-col gap-1">
                        <label className="text-xs text-zinc-500 font-medium">Replay Frequency</label>
                        <select
                            className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-700 outline-none focus:border-purple-400"
                            value={frequency}
                            onChange={(e) => updateData('frequency', e.target.value)}
                        >
                            {FREQUENCIES.map(f => (
                                <option key={f} value={f}>{f}</option>
                            ))}
                        </select>
                    </div>



                </div>
            </div>

            <Handle type="source" position={Position.Right} className="!bg-purple-500 !w-3 !h-3" />
        </div>
    );
};

export default memo(DataReplayerNode);
