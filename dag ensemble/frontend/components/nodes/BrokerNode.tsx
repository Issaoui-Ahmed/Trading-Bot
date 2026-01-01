import React, { memo, useState, useEffect, useCallback } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';

const BrokerNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const [datasets, setDatasets] = useState<string[]>([]);
    const stats = data.latestData || {};

    const pnl = stats.pnl || 0;
    const pnlColor = pnl >= 0 ? 'text-green-600' : 'text-red-600';

    // State for configuration
    const mode = data.mode || 'live';
    const pair = data.pair || 'XBTUSD';
    const selectedDataset = data.dataset || '';
    const initialCapital = data.initialCapital || 10000;

    const lastFill = stats.last_fill || {};

    useEffect(() => {
        fetch('http://localhost:8001/datasets')
            .then(res => res.json())
            .then(data => setDatasets(data.datasets || []))
            .catch(err => console.error("Failed to fetch datasets", err));
    }, []);

    const updateData = useCallback((key: string, value: any) => {
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
        <div className={`p-4 border rounded-lg bg-white shadow-md w-80 transition-all duration-200 ${data.isExecuting ? 'border-indigo-500 ring-2 ring-indigo-200' : 'border-indigo-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-indigo-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-indigo-100 flex items-center justify-center text-indigo-600 font-bold text-xs">BK</div>
                    <div className="font-bold text-sm text-zinc-900">Broker Simulator</div>
                </div>
                {data.isExecuting && (
                    <div className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
                )}
            </div>

            <div className="space-y-3">
                {/* Status / Error Message */}
                {stats.status && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-800 text-[10px] p-2 rounded">
                        <strong>Status:</strong> {stats.status}
                    </div>
                )}

                {/* Order Execution Info */}
                <div className="bg-white p-2 rounded border border-indigo-100 text-xs shadow-sm">
                    <div className="text-[10px] text-indigo-400 uppercase font-semibold mb-1">Order Execution</div>
                    {stats.action && stats.action !== 'pass' ? (
                        <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                            <div className="text-zinc-500">Side:</div>
                            <div className={`font-bold ${stats.action === 'buy' ? 'text-green-600' : 'text-red-500'}`}>{stats.action.toUpperCase()}</div>

                            <div className="text-zinc-500">Price:</div>
                            <div className="font-mono">${stats.price?.toFixed(2)}</div>

                            <div className="text-zinc-500">Volume:</div>
                            <div className="font-mono">{stats.volume?.toFixed(6)}</div>

                            <div className="text-zinc-500">Fee:</div>
                            <div className="font-mono text-amber-600">${stats.fee?.toFixed(4)}</div>

                            {stats.cost && (
                                <>
                                    <div className="text-zinc-500">Total Cost:</div>
                                    <div className="font-mono font-bold">${stats.cost?.toFixed(2)}</div>
                                </>
                            )}

                            {stats.revenue && (
                                <>
                                    <div className="text-zinc-500">Total Rev:</div>
                                    <div className="font-mono font-bold text-green-600">${stats.revenue?.toFixed(2)}</div>
                                </>
                            )}

                            <div className="text-zinc-500 col-span-2 mt-1 pt-1 border-t border-dashed border-zinc-200 text-center">
                                {stats.timestamp ? new Date(stats.timestamp * 1000).toLocaleTimeString() : ''}
                            </div>
                        </div>
                    ) : (
                        <div className="text-zinc-400 text-center italic py-2">
                            {stats.action === 'pass' ? 'Action: PASS' : 'Waiting for Order...'}
                        </div>
                    )}
                </div>

                {/* History List */}
                {stats.history && stats.history.length > 0 && (
                    <div className="mt-2">
                        <div className="text-[10px] text-zinc-400 font-semibold uppercase mb-1">Recent History</div>
                        <div className="max-h-32 overflow-y-auto border border-zinc-200 rounded bg-zinc-50 text-[10px]">
                            <table className="w-full text-left">
                                <thead className="bg-zinc-100 text-zinc-500 sticky top-0">
                                    <tr>
                                        <th className="p-1">Time</th>
                                        <th className="p-1">Side</th>
                                        <th className="p-1">Price</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stats.history.slice().reverse().map((order: any, idx: number) => (
                                        <tr key={idx} className="border-b border-zinc-100 last:border-0 hover:bg-white">
                                            <td className="p-1 text-zinc-600">
                                                {new Date(order.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                            </td>
                                            <td className={`p-1 font-bold ${order.action === 'buy' ? 'text-green-600' : 'text-red-600'}`}>
                                                {order.action.toUpperCase()}
                                            </td>
                                            <td className="p-1 font-mono text-zinc-700">
                                                {order.price?.toFixed(2)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Configuration Panel */}
                <div className="border-t border-zinc-100 pt-3">
                    <div className="text-[10px] text-zinc-400 font-semibold uppercase mb-2">Configuration</div>

                    {/* Simulation Settings */}
                    <div className="flex flex-col gap-1">
                        <label className="text-xs text-zinc-500 font-medium">Simulation Dataset</label>
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

                    <div className="flex flex-col gap-1 mt-2">
                        <label className="text-xs text-zinc-500 font-medium">Initial Capital</label>
                        <input
                            type="number"
                            className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-700 outline-none focus:border-indigo-400"
                            value={initialCapital}
                            onChange={(e) => updateData('initialCapital', parseFloat(e.target.value))}
                        />
                    </div>
                </div>
            </div>

            <Handle type="source" position={Position.Right} className="!bg-indigo-500 !w-3 !h-3" />
        </div>
    );
};

export default memo(BrokerNode);
