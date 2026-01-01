import React, { useEffect, useState } from 'react';
import { getWorkflows, saveWorkflowLink, loadWorkflow, deleteWorkflow } from '../utils/api';
import { Node, Edge } from 'reactflow';

interface SidebarProps {
    nodes: Node[];
    edges: Edge[];
    setNodes: (nodes: Node[]) => void;
    setEdges: (edges: Edge[]) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ nodes, edges, setNodes, setEdges }) => {
    const [workflowName, setWorkflowName] = useState('');
    const [currentWorkflow, setCurrentWorkflow] = useState<string | null>(null);
    const [savedWorkflows, setSavedWorkflows] = useState<string[]>([]);

    useEffect(() => {
        refreshWorkflows();
    }, []);

    const refreshWorkflows = async () => {
        try {
            const data = await getWorkflows();
            if (data.workflows) {
                setSavedWorkflows(data.workflows);
            }
        } catch (e) {
            console.error("Failed to load workflows", e);
        }
    };

    const handleNewWorkflow = () => {
        if (nodes.length > 0 && !confirm("Create new workflow? Unsaved changes will be lost.")) return;
        setNodes([]);
        setEdges([]);
        setWorkflowName('');
        setCurrentWorkflow(null);
    };

    const handleSave = async () => {
        const nameToSave = currentWorkflow || workflowName;
        if (!nameToSave) {
            alert("Please enter a workflow name");
            return;
        }

        try {
            // Simplified node/edge data
            const cleanNodes = nodes.map(n => ({
                id: n.id,
                type: n.type,
                position: n.position,
                data: n.data // Ensure data is serializable
            }));
            const cleanEdges = edges.map(e => ({
                id: e.id,
                source: e.source,
                target: e.target,
                sourceHandle: e.sourceHandle,
                targetHandle: e.targetHandle
            }));

            await saveWorkflowLink(nameToSave, cleanNodes, cleanEdges);

            if (!currentWorkflow) {
                setCurrentWorkflow(nameToSave);
                setWorkflowName('');
            }
            refreshWorkflows();
            alert(`Workflow "${nameToSave}" saved!`);
        } catch (e) {
            console.error(e);
            alert('Failed to save');
        }
    };

    const handleLoad = async (name: string) => {
        if (nodes.length > 0 && !confirm(`Load workflow "${name}"? Unsaved changes will be lost.`)) return;
        try {
            const data = await loadWorkflow(name);
            if (data && data.nodes) {
                setNodes(data.nodes);
                setEdges(data.edges || []);
                setCurrentWorkflow(name);
                setWorkflowName(''); // Clear input as we are now in "edit mode" of `name`
            }
        } catch (e) {
            console.error(e);
            alert('Failed to load');
        }
    };

    const handleDelete = async (name: string) => {
        if (!confirm(`Are you sure you want to delete "${name}"?`)) return;
        try {
            await deleteWorkflow(name);
            if (currentWorkflow === name) {
                setCurrentWorkflow(null);
                setNodes([]);
                setEdges([]);
            }
            refreshWorkflows();
        } catch (e) {
            console.error(e);
            alert('Failed to delete');
        }
    };

    const onDragStart = (event: React.DragEvent, nodeType: string) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    };

    return (
        <aside className="w-64 bg-white border-r border-gray-200 p-4 flex flex-col gap-4 h-full overflow-y-auto">
            <div className="mb-4">
                <div className="flex justify-between items-center">
                    <h1 className="text-xl font-bold text-gray-800">ML Workflow</h1>
                    <button
                        onClick={handleNewWorkflow}
                        className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-600 px-2 py-1 rounded border border-gray-300 transition-colors"
                        title="New Workflow"
                    >
                        New
                    </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">Drag nodes to the canvas</p>
            </div>

            <div className="space-y-4">
                <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">Triggers</h3>
                    <div
                        className="p-3 bg-red-50 border border-red-200 rounded cursor-grab flex items-center gap-2 hover:bg-red-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'liveDataFeed')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-red-500"></div>
                        <span className="text-sm font-medium text-gray-700">Live Data Feed</span>
                    </div>
                    <div
                        className="p-3 bg-purple-50 border border-purple-200 rounded cursor-grab flex items-center gap-2 hover:bg-purple-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'dataReplayer')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                        <span className="text-sm font-medium text-gray-700">Data Replayer</span>
                    </div>
                </div>

                <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">Storage</h3>
                    <div
                        className="p-3 bg-cyan-50 border border-cyan-200 rounded cursor-grab flex items-center gap-2 hover:bg-cyan-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'dataStore')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-cyan-500"></div>
                        <span className="text-sm font-medium text-gray-700">Write & Fetch</span>
                    </div>
                    <div
                        className="p-3 bg-fuchsia-50 border border-fuchsia-200 rounded cursor-grab flex items-center gap-2 hover:bg-fuchsia-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'readNode')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-fuchsia-500"></div>
                        <span className="text-sm font-medium text-gray-700">Read & View</span>
                    </div>
                </div>

                <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">Processing</h3>
                    <div
                        className="p-3 bg-indigo-50 border border-indigo-200 rounded cursor-grab flex items-center gap-2 hover:bg-indigo-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'featureEngineering')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-indigo-500"></div>
                        <span className="text-sm font-medium text-gray-700">Feature Engineering</span>
                    </div>
                    <div
                        className="p-3 bg-orange-50 border border-orange-200 rounded cursor-grab flex items-center gap-2 hover:bg-orange-100 transition-colors"
                        onDragStart={(event) => onDragStart(event, 'mergeNode')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                        <span className="text-sm font-medium text-gray-700">Merge Node</span>
                    </div>
                    <div
                        className="p-3 bg-yellow-50 border border-yellow-200 rounded cursor-grab flex items-center gap-2 hover:bg-yellow-100 transition-colors"
                        onDragStart={(event) => onDragStart(event, 'ifElseNode')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                        <span className="text-sm font-medium text-gray-700">If / Else</span>
                    </div>
                </div>

                <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">Models</h3>
                    <div
                        className="p-3 bg-blue-50 border border-blue-200 rounded cursor-grab flex items-center gap-2 hover:bg-blue-100 transition-colors"
                        onDragStart={(event) => onDragStart(event, 'model')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                        <span className="text-sm font-medium text-gray-700">Predict</span>
                    </div>
                    <div
                        className="p-3 bg-purple-50 border border-purple-200 rounded cursor-grab flex items-center gap-2 hover:bg-purple-100 transition-colors mt-2"
                        onDragStart={(event) => onDragStart(event, 'trainNode')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                        <span className="text-sm font-medium text-gray-700">Train</span>
                    </div>
                </div>

                <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">Execution</h3>
                    <div
                        className="p-3 bg-orange-50 border border-orange-200 rounded cursor-grab flex items-center gap-2 hover:bg-orange-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'tradingBrain')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                        <span className="text-sm font-medium text-gray-700">Trading Brain</span>
                    </div>
                    <div
                        className="p-3 bg-purple-50 border border-purple-200 rounded cursor-grab flex items-center gap-2 hover:bg-purple-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'randomAction')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                        <span className="text-sm font-medium text-gray-700">Random Action</span>
                    </div>
                    <div
                        className="p-3 bg-indigo-50 border border-indigo-200 rounded cursor-grab flex items-center gap-2 hover:bg-indigo-100 transition-colors mb-2"
                        onDragStart={(event) => onDragStart(event, 'brokerNode')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-indigo-500"></div>
                        <span className="text-sm font-medium text-gray-700">Broker Simulator</span>
                    </div>
                    <div
                        className="p-3 bg-fuchsia-50 border border-fuchsia-200 rounded cursor-grab flex items-center gap-2 hover:bg-fuchsia-100 transition-colors"
                        onDragStart={(event) => onDragStart(event, 'evalNode')}
                        draggable
                    >
                        <div className="w-2 h-2 rounded-full bg-fuchsia-500"></div>
                        <span className="text-sm font-medium text-gray-700">Eval Node</span>
                    </div>
                </div>
            </div>

            <div className="border-t border-gray-200 mt-4 pt-4">
                <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">Workflow Management</h3>

                {/* Save Controls */}
                <div className="flex gap-2 mb-3">
                    {currentWorkflow ? (
                        <div className="w-full flex items-center justify-between text-sm bg-blue-50 text-blue-800 px-2 py-1 rounded border border-blue-200">
                            <span className="font-semibold truncate max-w-[100px]" title={currentWorkflow}>{currentWorkflow}</span>
                            <button
                                onClick={handleSave}
                                className="text-[10px] bg-blue-500 text-white px-2 py-0.5 rounded hover:bg-blue-600"
                            >
                                Save
                            </button>
                        </div>
                    ) : (
                        <>
                            <input
                                type="text"
                                value={workflowName}
                                onChange={(e) => setWorkflowName(e.target.value)}
                                placeholder="Name..."
                                className="w-full text-sm border border-gray-300 rounded px-2 py-1"
                            />
                            <button
                                onClick={handleSave}
                                className="bg-green-500 text-white text-xs px-2 py-1 rounded hover:bg-green-600 whitespace-nowrap"
                            >
                                Save As
                            </button>
                        </>
                    )}
                </div>

                {/* Workflow List */}
                <div className="max-h-48 overflow-y-auto space-y-1">
                    {savedWorkflows.map(name => (
                        <div key={name} className={`flex justify-between items-center text-sm p-1.5 rounded group ${currentWorkflow === name ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50 border border-transparent'}`}>
                            <span
                                className={`truncate w-24 cursor-pointer ${currentWorkflow === name ? 'font-semibold text-blue-700' : 'text-gray-700'}`}
                                onClick={() => handleLoad(name)}
                                title={name}
                            >
                                {name}
                            </span>
                            <div className="flex items-center gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                                <button
                                    onClick={() => handleDelete(name)}
                                    className="text-gray-400 hover:text-red-500 p-1"
                                    title="Delete"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    ))}
                    {savedWorkflows.length === 0 && <p className="text-gray-400 text-xs italic text-center py-2">No saved workflows</p>}
                </div>
            </div>

            <div className="mt-auto p-4 bg-gray-50 rounded text-[10px] text-gray-400">
                v1.3.0 • Inference Mode
            </div>
        </aside >
    );
};

export default Sidebar;
