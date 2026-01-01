import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, NodeProps, useReactFlow, useUpdateNodeInternals } from 'reactflow';
import Editor from '@monaco-editor/react';

const IfElseNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const updateNodeInternals = useUpdateNodeInternals();
    const [code, setCode] = useState(data.code || "return True");

    useEffect(() => {
        updateNodeInternals(id);
    }, [id, updateNodeInternals]);

    const handleEditorChange = (value: string | undefined) => {
        const val = value || "";
        setCode(val);
        setNodes((nds) =>
            nds.map((node) => {
                if (node.id === id) {
                    return { ...node, data: { ...node.data, code: val } };
                }
                return node;
            })
        );
    };

    const handleEditorDidMount = (editor: any, monaco: any) => {
        // Stop propagation of key events to prevent React Flow from capturing them
        // This is crucial for the Space key which React Flow might treat as a pan shortcut
        const domNode = editor.getDomNode();
        if (domNode) {
            domNode.addEventListener('keydown', (e: KeyboardEvent) => e.stopPropagation());
        }
    };

    const latestResult = data.latestData?.condition_met;

    return (
        <div className={`relative p-4 border rounded-lg bg-white shadow-md w-72 transition-all duration-200 ${data.isExecuting ? 'border-yellow-500 ring-2 ring-yellow-200' : 'border-yellow-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-yellow-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-yellow-100 flex items-center justify-center text-yellow-600 font-bold text-xs">IF</div>
                    <div className="font-bold text-sm text-zinc-900">If / Else</div>
                </div>
                {/* Execution Badge */}
                {data.isExecuting && (
                    <div className="w-2 h-2 rounded-full bg-yellow-500 animate-ping" />
                )}
            </div>

            <div className="space-y-3">
                <div>
                    <label className="text-[10px] text-zinc-500 font-semibold uppercase block mb-1">Condition (Python)</label>
                    <div
                        className="border border-zinc-200 rounded overflow-hidden nodrag nopan nowheel"
                        onKeyDown={(e) => { e.stopPropagation(); }}
                        onMouseDown={(e) => { e.stopPropagation(); }}
                    >
                        <Editor
                            height="120px"
                            defaultLanguage="python"
                            value={code}
                            onChange={handleEditorChange}
                            onMount={handleEditorDidMount}
                            theme="light"
                            options={{
                                minimap: { enabled: false },
                                lineNumbers: 'off',
                                folding: false,
                                fontSize: 11,
                                padding: { top: 8 },
                                scrollBeyondLastLine: false,
                                overviewRulerLanes: 0,
                                hideCursorInOverviewRuler: true,
                                renderLineHighlight: 'none',
                            }}
                        />
                    </div>
                    {data.latestData?.error && (
                        <div className="mt-1 p-1.5 bg-red-50 border border-red-200 rounded text-[9px] text-red-600 font-mono leading-tight break-words">
                            <span className="font-bold">Error: </span>{data.latestData.error}
                        </div>
                    )}
                    <p className="text-[9px] text-zinc-400 mt-1">Input available as `data_package`</p>
                </div>

                <div className="bg-zinc-50 border border-zinc-100 rounded p-2 flex justify-between items-center">
                    <span className="text-[10px] text-zinc-400 uppercase font-semibold">Result</span>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${latestResult === true ? 'bg-green-100 text-green-700' : latestResult === false ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-500'}`}>
                        {latestResult === undefined ? 'Pending' : (latestResult ? 'TRUE' : 'FALSE')}
                    </span>
                </div>
            </div>

            {/* Outputs */}
            <div className="absolute -right-8 top-[35%] -translate-y-1/2 text-[9px] text-zinc-400 font-semibold">True</div>
            <Handle
                id="true"
                type="source"
                position={Position.Right}
                className="!bg-green-500 !w-3 !h-3"
                style={{ top: '35%' }}
            />

            <div className="absolute -right-8 top-[65%] -translate-y-1/2 text-[9px] text-zinc-400 font-semibold">False</div>
            <Handle
                id="false"
                type="source"
                position={Position.Right}
                className="!bg-red-500 !w-3 !h-3"
                style={{ top: '65%' }}
            />
        </div>
    );
};

export default memo(IfElseNode);
