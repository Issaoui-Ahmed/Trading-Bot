const API_URL = 'http://localhost:8001';

export const getModels = async () => {
    const res = await fetch(`${API_URL}/models`);
    return res.json();
};

export const getModelMetadata = async (modelName: string) => {
    const res = await fetch(`${API_URL}/models/${modelName}/metadata`);
    if (!res.ok) return null;
    return res.json();
};

export const getTrainingScripts = async () => {
    const res = await fetch(`${API_URL}/training-scripts`);
    return res.json();
};

export const getStreamData = async () => {
    const res = await fetch(`${API_URL}/stream`);
    return res.json();
};

export const updateWorkflow = async (nodes: any[], edges: any[]) => {
    const res = await fetch(`${API_URL}/workflow`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges }),
    });
    return res.json();
};

export const getWorkflows = async () => {
    const res = await fetch(`${API_URL}/workflows`);
    return res.json();
};

export const saveWorkflowLink = async (name: string, nodes: any[], edges: any[]) => {
    const res = await fetch(`${API_URL}/workflows/${name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges }),
    });
    return res.json();
};

export const loadWorkflow = async (name: string) => {
    const res = await fetch(`${API_URL}/workflows/${name}`);
    if (!res.ok) return null;
    return res.json();
    return res.json();
};

export const deleteWorkflow = async (name: string) => {
    const res = await fetch(`${API_URL}/workflows/${name}`, {
        method: 'DELETE',
    });
    if (!res.ok) return null;
    return res.json();
};

// Deprecated or Unused placeholders if needed to prevent immediate crashes before full refactor
export const getDataset = async () => ({ columns: [] });
export const getDataCenterFeatures = async () => ({ features: [] });
export const trainNode = async () => ({ mse: 0 });
export const runInference = async () => ({ prediction: 0 });

export const pauseWorkflow = async () => {
    const res = await fetch(`${API_URL}/workflow/pause`, { method: 'POST' });
    return res.json();
};

export const resumeWorkflow = async () => {
    const res = await fetch(`${API_URL}/workflow/resume`, { method: 'POST' });
    return res.json();
};

export const resetWorkflow = async () => {
    const res = await fetch(`${API_URL}/workflow/reset`, { method: 'POST' });
    return res.json();
};

export { API_URL };
