import React, { memo, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';

const EvalNode = ({ data, id }: NodeProps) => {
    // Data from backend evalNode
    // Structure: { distribution: [{range_start, range_end, count}], metrics: {win_rate, avg_return, total_trades} }
    const results = data.latestData || {};
    const distribution = results.distribution || [];
    const metrics = results.metrics || {};

    const chartData = useMemo(() => {
        return distribution.map((bin: any) => ({
            range: `${(bin.range_start * 100).toFixed(2)}%`,
            count: bin.count,
            val: bin.range_start
        }));
    }, [distribution]);

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-96 transition-all duration-200 ${data.isExecuting ? 'border-fuchsia-500 ring-2 ring-fuchsia-200' : 'border-fuchsia-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-fuchsia-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-fuchsia-100 flex items-center justify-center text-fuchsia-600 font-bold text-xs">EV</div>
                    <div className="font-bold text-sm text-zinc-900">Evaluation</div>
                </div>
                {data.isExecuting && (
                    <div className="w-2 h-2 rounded-full bg-fuchsia-500 animate-ping" />
                )}
            </div>

            <div className="space-y-3">
                {/* Metrics Panel */}
                <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="bg-zinc-50 p-2 rounded border border-zinc-100 text-center">
                        <div className="text-[10px] text-zinc-400 uppercase">Trades</div>
                        <div className="font-mono font-bold text-zinc-700">
                            {metrics.total_trades ?? 0}
                        </div>
                    </div>
                    <div className="bg-zinc-50 p-2 rounded border border-zinc-100 text-center">
                        <div className="text-[10px] text-zinc-400 uppercase">Win Rate</div>
                        <div className={`font-mono font-bold ${(metrics.win_rate ?? 0) >= 0.5 ? 'text-green-600' : 'text-zinc-700'}`}>
                            {((metrics.win_rate ?? 0) * 100).toFixed(1)}%
                        </div>
                    </div>
                    <div className="bg-zinc-50 p-2 rounded border border-zinc-100 text-center">
                        <div className="text-[10px] text-zinc-400 uppercase">Avg Ret</div>
                        <div className={`font-mono font-bold ${(metrics.avg_return ?? 0) >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                            {((metrics.avg_return ?? 0) * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>

                {/* Histogram */}
                <div className="bg-white p-2 rounded border border-zinc-100 h-40">
                    <div className="text-[10px] text-zinc-400 uppercase mb-1 flex justify-between">
                        <span>Return Distribution</span>
                        <span className="text-[9px] text-zinc-300">{(metrics.total_trades ?? 0) === 0 ? 'No Trades Yet' : ''}</span>
                    </div>
                    {chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f4f4f5" />
                                <XAxis dataKey="range" fontSize={9} tick={{ fill: '#71717a' }} interval={1} />
                                <YAxis fontSize={9} tick={{ fill: '#71717a' }} allowDecimals={false} />
                                <Tooltip
                                    contentStyle={{ fontSize: '10px', padding: '4px', borderRadius: '4px', border: '1px solid #e4e4e7' }}
                                    cursor={{ fill: '#f4f4f5' }}
                                />
                                <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                                    {chartData.map((entry: any, index: number) => (
                                        <Cell key={`cell-${index}`} fill={entry.val >= 0 ? '#10b981' : '#ef4444'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="h-full flex items-center justify-center text-zinc-300 text-xs italic">
                            Waiting for trade data...
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default memo(EvalNode);
