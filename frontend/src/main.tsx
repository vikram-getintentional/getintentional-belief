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
<<<<<<< Updated upstream
=======
import ICP from './pages/ICP'; 
import ValueProp from './pages/ValueProposition';
import ReverseCaseStudies from './pages/ReverseCaseStudies'
import Arsenal from './pages/Arsenal.tsx'
import CRMUpload from './pages/CRMUpload.tsx'
import Admin from './pages/Admin.tsx'
import TargetAccounts from './pages/TargetAccounts.tsx'

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
          <Route path="/personas" element={<Personas/>} />
=======
          <Route path="/value-prop" element={<ValueProp />} />
          <Route path="/personas" element={<Personas />} />
          <Route path="/icp" element={<ICP />} />
          <Route path="/reverse-case-studies" element={<ReverseCaseStudies />} />
          <Route path="/arsenal" element={<Arsenal />} />
          <Route path="/crm-upload" element={<CRMUpload />} />
          <Route path="/target-accounts" element={<TargetAccounts />} />
          <Route path="/admin" element={<Admin />} />
          
          {/* Add other routes here */}
>>>>>>> Stashed changes

        </Routes>
      </AppLayout>
    </BrowserRouter>
  </React.StrictMode>
)
