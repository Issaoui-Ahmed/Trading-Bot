import React, { memo, useState, useEffect, useCallback, useMemo } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const PaperTradingNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const [datasets, setDatasets] = useState<string[]>([]);
    const stats = data.latestData || {};

    // Helper to color PnL
    const pnl = stats.pnl || 0;
    const pnlColor = pnl >= 0 ? 'text-green-600' : 'text-red-600';

    // State for configuration
    const mode = data.mode || 'live';
    const pair = data.pair || 'XBTUSD';
    const selectedDataset = data.dataset || '';
    const initialCapital = data.initialCapital || 10000;

    const tradeReturns = stats.trade_returns || [];
    const dailyReturns = stats.daily_returns || [];
    const barReturns = stats.bar_returns || [];

    // Binning Helper
    const computeHistogram = (data: number[], binCount: number = 10) => {
        if (!data || data.length === 0) return [];

        const min = Math.min(...data);
        const max = Math.max(...data);

        // Handle single value case
        if (min === max) {
            return [{
                range: min.toFixed(2) + '%',
                count: data.length,
                val: min
            }];
        }

        const step = (max - min) / binCount;
        const bins = Array(binCount).fill(0).map((_, i) => ({
            range: (min + i * step).toFixed(2), // + "%",
            count: 0,
            val: min + i * step
        }));

        data.forEach(v => {
            let idx = Math.floor((v - min) / step);
            if (idx >= binCount) idx = binCount - 1;
            bins[idx].count++;
        });

        return bins;
    };

    const tradeHist = useMemo(() => computeHistogram(tradeReturns.map((r: number) => r * 100)), [tradeReturns]);
    const dailyHist = useMemo(() => computeHistogram(dailyReturns.map((r: number) => r * 100)), [dailyReturns]);

    useEffect(() => {
        // Fetch available datasets for Replay mode
        fetch('http://localhost:8001/replay-datasets')
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
        <div className={`p-4 border rounded-lg bg-white shadow-md w-96 transition-all duration-200 ${data.isExecuting ? 'border-emerald-500 ring-2 ring-emerald-200' : 'border-emerald-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-emerald-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-emerald-100 flex items-center justify-center text-emerald-600 font-bold text-xs">PT</div>
                    <div className="font-bold text-sm text-zinc-900">Paper Trading</div>
                </div>
                {data.isExecuting && (
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                )}
            </div>

            <div className="space-y-3">
                {/* Stats Panel */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-zinc-50 p-2 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 uppercase">Cash</div>
                        <div className="font-mono font-bold text-zinc-700">
                            ${stats.cash?.toFixed(2) ?? '---'}
                        </div>
                    </div>
                    <div className="bg-zinc-50 p-2 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 uppercase">Holdings</div>
                        <div className="font-mono font-bold text-zinc-700">
                            {stats.holdings?.toFixed(4) ?? '---'}
                        </div>
                    </div>
                </div>

                <div className="bg-zinc-50 p-2 rounded border border-zinc-100 text-xs">
                    <div className="text-[10px] text-zinc-400 uppercase">Total Value</div>
                    <div className="font-mono font-bold text-zinc-800 text-lg">
                        ${stats.total_value?.toFixed(2) ?? '---'}
                    </div>
                    <div className={`font-mono font-bold text-xs mt-1 ${pnlColor}`}>
                        PnL: {pnl > 0 ? '+' : ''}{pnl.toFixed(2)}
                    </div>
                </div>

                {stats.last_action && stats.last_action !== 'pass' && stats.last_action !== 'none' && (
                    <div className="bg-emerald-50 text-emerald-700 text-[10px] px-2 py-1 rounded border border-emerald-100 text-center font-semibold">
                        Last Trade: {stats.last_action.toUpperCase()}
                    </div>
                )}

                {/* Distributions */}
                <div className="grid grid-cols-1 gap-2">
                    <div className="bg-zinc-50 p-2 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 uppercase mb-1">Return / Trade Distrib.</div>
                        <div className="h-24 w-full">
                            {tradeHist.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={tradeHist}>
                                        <XAxis dataKey="range" fontSize={8} tick={true} interval={0} />
                                        <Tooltip
                                            contentStyle={{ fontSize: '10px', padding: '2px' }}
                                            formatter={(value: any) => [value, 'Count']}
                                            labelFormatter={(l) => `${l}%`}
                                        />
                                        <Bar dataKey="count" fill="#10b981" radius={[2, 2, 0, 0]}>
                                            {tradeHist.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.val >= 0 ? '#10b981' : '#ef4444'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="h-full flex items-center justify-center text-[10px] text-zinc-400">No Trades Yet</div>
                            )}
                        </div>
                    </div>

                    <div className="bg-zinc-50 p-2 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 uppercase mb-1">Daily Return Distrib.</div>
                        <div className="h-24 w-full">
                            {dailyHist.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={dailyHist}>
                                        <XAxis dataKey="range" fontSize={8} tick={true} interval={0} />
                                        <Tooltip
                                            contentStyle={{ fontSize: '10px', padding: '2px' }}
                                            formatter={(value: any) => [value, 'Count']}
                                            labelFormatter={(l) => `${l}%`}
                                        />
                                        <Bar dataKey="count" fill="#8b5cf6" radius={[2, 2, 0, 0]}>
                                            {dailyHist.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.val >= 0 ? '#8b5cf6' : '#ef4444'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="h-full flex items-center justify-center text-[10px] text-zinc-400">No Daily Data Yet</div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Configuration Panel */}
                <div className="border-t border-zinc-100 pt-3">
                    <div className="text-[10px] text-zinc-400 font-semibold uppercase mb-2">Configuration</div>

                    {/* Mode Selector */}
                    <div className="flex bg-zinc-100 rounded p-1 mb-2">
                        <button
                            className={`flex-1 text-[10px] font-bold py-1 rounded ${mode === 'live' ? 'bg-white shadow text-emerald-600' : 'text-zinc-500'}`}
                            onClick={() => updateData('mode', 'live')}
                        >
                            LIVE
                        </button>
                        <button
                            className={`flex-1 text-[10px] font-bold py-1 rounded ${mode === 'replay' ? 'bg-white shadow text-purple-600' : 'text-zinc-500'}`}
                            onClick={() => updateData('mode', 'replay')}
                        >
                            REPLAY
                        </button>
                    </div>

                    {mode === 'live' ? (
                        <div className="flex flex-col gap-1">
                            <label className="text-xs text-zinc-500 font-medium">Pair</label>
                            <input
                                type="text"
                                className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-700 outline-none focus:border-emerald-400"
                                value={pair}
                                onChange={(e) => updateData('pair', e.target.value)}
                                placeholder="e.g. XBTUSD"
                            />
                        </div>
                    ) : (
                        <div className="flex flex-col gap-1">
                            <label className="text-xs text-zinc-500 font-medium">Replay Dataset</label>
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
                    )}

                    <div className="flex flex-col gap-1 mt-2">
                        <label className="text-xs text-zinc-500 font-medium">Initial Capital</label>
                        <input
                            type="number"
                            className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-700 outline-none focus:border-emerald-400"
                            value={initialCapital}
                            onChange={(e) => updateData('initialCapital', parseFloat(e.target.value))}
                        />
                    </div>
                </div>
            </div>

            {/* Output for metrics downstream if needed, though usually this is a sink node */}
            <Handle type="source" position={Position.Right} className="!bg-emerald-500 !w-3 !h-3" />
        </div>
    );
};

export default memo(PaperTradingNode);
