import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from "react"; 
import { Menu } from "lucide-react";
import { cn } from '../../utils/cn'; // optional: or just use template strings

const navSections = [
    {
      title: 'Execution',
      items: [
        { name: 'Dashboard', path: '/dashboard' },
        { name: 'Marketing Planner', path: '/marketing-planner' },
        { name: 'Insights Inbox', path: '/insights-inbox' },
      ],
    },
    {
      title: 'Library',
      items: [
        { name: 'Value Prop', path: '/value-prop' },
        { name: 'Personas', path: '/personas' },
        { name: 'ZMOT', path: '/zmot' },
        { name: 'Arsenal', path: '/arsenal' },
      ],
    },
    {
      title: 'Geek Out',
      items: [
        { name: 'Models', path: '/models' },
        { name: 'Reverse Case Studies', path: '/reverse-case-studies' },
      ],
    },
    {
      title: 'Setup',
      items: [
        { name: 'CRM Upload', path: '/crm-upload' },
        { name: 'Engagements Setup', path: '/engagements-setup' },
        { name: 'Target Accounts', path: '/target-accounts' },
      ],
    },
    {
      title: 'Other',
      items: [
        { name: 'Admin', path: '/admin' },
        { name: 'Help', path: '/help' },
      ],
    },
  ];

  function AppLayout({ children }: { children: React.ReactNode }) {
    const location = useLocation();
    const [sidebarOpen, setSidebarOpen] = useState(() => {
      if (typeof window !== "undefined") {
        return window.innerWidth >= 1024;
      }
      return true;
    });
  
    useEffect(() => {
      const handleResize = () => {
        if (window.innerWidth < 1024) setSidebarOpen(false);
        else setSidebarOpen(true);
      };
  
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }, []);
  
    return (
      <div className="flex min-h-screen bg-gray-50 text-gray-800 w-full">
        {/* Sidebar toggle button */}
        <button
          className="fixed top-4 left-4 z-50 p-2 bg-white border rounded shadow"
          onClick={() => setSidebarOpen((prev) => !prev)}
        >
          <Menu className="h-5 w-5" />
        </button>
  
        {/* Sidebar */}
        <aside
          className={`fixed lg:static top-0 left-0 h-full z-40 w-64 bg-white px-4 py-6 space-y-8 overflow-y-auto transform transition-transform duration-200 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
        >
          <div className="flex items-center gap-2">
            <img
              src="/assets/getintentional-logo.png"
              alt="GetIntentional logo"
              className="max-h-10 w-auto mx-auto"
            />
          </div>
          {navSections.map((section) => (
            <div key={section.title}>
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                {section.title}
              </h2>
              <nav className="space-y-1">
                {section.items.map((item) => (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`block pl-4 py-2 rounded-md text-sm font-medium transition ${
                      location.pathname === item.path
                        ? "bg-indigo-100 text-indigo-700"
                        : "hover:bg-gray-100"
                    }`}
                  >
                    {item.name}
                  </Link>
                ))}
              </nav>
            </div>
          ))}
        </aside>
  
        {/* Main content */}
        <main className="flex-1 p-10 ml-0 transition-all duration-200 ease-in-out">
          {children}
        </main>
      </div>
    );
  }
  
  export default AppLayout;