class AgentContext:
    def __init__(self, max_depth=3):
        self.visited_jobs = set()
        self.current_depth = 0
        self.max_depth = max_depth
        self.hop_plus_cache = []
        self.zmot_cache = []
        self.pain_source_cache = []
        self.inference_log = {}
    
    def mark_visited(self, job_id):
        self.visited_jobs.add(job_id)

    def is_visited(self, job_id):
        return job_id in self.visited_jobs

    def log(self, job_id, source, data):
        self.inference_log[job_id] = {"source": source, "data": data}
