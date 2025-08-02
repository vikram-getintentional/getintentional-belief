import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import "./output.css";
import Onboarding from "./pages/Onboarding";
import AppLayout from "./layout/AppLayout";
import Personas from "./pages/Personas";
import Zmot from "./pages/Zmot";
import ValueProp from "./pages/ValueProposition";
import "./output.css";
import {
  websocketService,
  WebSocketMessage,
} from "./services/websocketService";
import * as Api from "./api";
import { taskService, TaskStatus } from "./services/taskService";

type Session = { token: string };
type Props = { token: string | null };
type Persona = { name: string };
function App({ token }: Props) {
  const [session, setSession] = useState<Session | null>(
    token ? { token } : null,
  );
  const [personaGenerationProgress, setPersonaGenerationProgress] = useState(0);
  const [personaTaskID, setPersonaTaskID] = useState<string | null>(null);
  const [suggestedPersonas, setSuggestedPersonas] = useState<Persona[]>([]);
  useEffect(() => {
    (async function (session, personaTaskID) {
      const token = session?.token;
      if (!token) return;
      const { companyID } = await Api.me(token);
      // We dont have websocket pushing celery task
      // status as of now. We stick to HTTP client
      // side polling
      websocketService.connect(companyID, token);
      // Listen for task completion events
      const handleTaskComplete = (message: WebSocketMessage) => {
        console.log(
          "✅ Received WebSocket notification: Task completed",
          message.data,
        );
        // if (message.product_id === product_id && message.event === "hop0_generation_complete") {
        //   handleTaskSuccess(message.data);
        // }
      };
      const handleTaskFailed = (message: WebSocketMessage) => {
        console.log(
          "❌ Received WebSocket notification: Task failed",
          message.error,
        );
      };
      websocketService.on("hop0_generation_complete", handleTaskComplete);
      websocketService.on("hop0_generation_failed", handleTaskFailed);

      if (personaTaskID) {
        const finalStatus = await taskService.pollTaskStatus(
          personaTaskID,
          token,
          (status: TaskStatus) => {
            console.log("📊 Task status update:", status);
            setPersonaGenerationProgress(status.progress);
            if (status.state === "SUCCESS") {
              setSuggestedPersonas(status.result.hop_0_results.personas);
            }
          },
        );
        console.log("finalStatus", finalStatus);
	setPersonaTaskID(null);
      }
    })(session, personaTaskID);
  }, [session, personaTaskID]);
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/onboarding" element={<Onboarding />} />
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login setSession={setSession} />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/value-prop" element={<ValueProp />} />
          <Route
            path="/personas"
            element={
              <Personas
                suggestedPersonas={suggestedPersonas}
                progress={personaGenerationProgress}
                token={session?.token}
                setPersonaTaskID={setPersonaTaskID}
              />
            }
          />
          <Route path="/zmot" element={<Zmot />} />
          {/* Add other routes here */}
        </Routes>
      </AppLayout>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <App token={localStorage.getItem("token")} />,
);
