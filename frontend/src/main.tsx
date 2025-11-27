import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Dashboard from './pages/Dashboard'
import './output.css';
import Onboarding from './pages/Onboarding';
import AppLayout from './layout/AppLayout';
import Personas from './pages/Personas'; 
import ICP from './pages/ICP'; 
import ValueProp from './pages/ValueProposition';
import ReverseCaseStudies from './pages/ReverseCaseStudies'
import RcsSimulator from './pages/RCSSimulator.tsx'
import Arsenal from './pages/Arsenal.tsx'
import CRMUpload from './pages/CRMUpload.tsx'
import Admin from './pages/Admin.tsx'
import TargetAccounts from './pages/TargetAccounts.tsx'
import Models from './pages/models.tsx'
import MarketingPlanner from './pages/MarketingPlanner.tsx'
import PainsMap from './pages/PainsMap.tsx'
import EngagementsSetup from './pages/Engagements.tsx'

import './output.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/onboarding" element={<Onboarding />} />
          <Route path="/" element={<App />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/value-prop" element={<ValueProp />} />
          <Route path="/personas" element={<Personas />} />
          <Route path="/icp" element={<ICP />} />
          <Route path="/reverse-case-studies" element={<ReverseCaseStudies />} />
          <Route path="/arsenal" element={<Arsenal />} />
          <Route path="/crm-upload" element={<CRMUpload />} />
          <Route path="/target-accounts" element={<TargetAccounts />} />
          <Route path="/models" element={<Models />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="/marketing-planner" element={<MarketingPlanner />} />
          <Route path="/rcs-simulator" element={<RcsSimulator />} />
          <Route path="/pains-map" element={<PainsMap />} />
          <Route path="/engagements-setup" element={<EngagementsSetup />} />

          {/* Add other routes here */}

        </Routes>
      </AppLayout>
    </BrowserRouter>
  </React.StrictMode>
)
