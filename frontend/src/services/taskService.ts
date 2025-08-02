export type TaskStatus = {
  task_id: string;
  state: "PENDING" | "PROCESSING" | "SUCCESS" | "FAILURE";
  message: string;
  progress: number;
  product_id: string;
  result: {
    hop_0_results: {
      capability_map: any,
      personas: any[],
    };
    status: string;
    product_id: string;
    company_id: string;
  };
  error?: string;
};

export type TaskResult = {
  status: string;
  product_id: string;
  company_id: string;
  hop_0_results: any;
  message: string;
};

class TaskService {
  private baseUrl = "http://localhost:8000";

  async startDeepAnalysis(
    productId: string,
    token: string,
  ): Promise<{ task_id: string; status: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/analyze/deep`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ product_id: productId }),
    });

    if (!response.ok) {
      throw new Error(`Failed to start analysis: ${response.statusText}`);
    }

    return response.json();
  }

  async getTaskStatus(taskId: string, token: string): Promise<TaskStatus> {
    const response = await fetch(`${this.baseUrl}/tasks/${taskId}/status`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get task status: ${response.statusText}`);
    }

    return response.json();
  }

  async pollTaskStatus(
    taskId: string,
    token: string,
    onUpdate?: (status: TaskStatus) => void,
    pollInterval = 2000,
  ): Promise<TaskStatus> {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getTaskStatus(taskId, token);

          if (onUpdate) {
            onUpdate(status);
          }

          if (status.state === "SUCCESS") {
            resolve(status);
          } else if (status.state === "FAILURE") {
            reject(new Error(status.error || status.message));
          } else {
            // Continue polling for PENDING or PROCESSING
            setTimeout(poll, pollInterval);
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }
}

export const taskService = new TaskService();
