import { ReactNode, useCallback, useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu } from "lucide-react";
import SidebarActions from "../components/SidebarActions";
import { cn } from '../../utils/cn'; // optional: or just use template strings

const navSections = [
    {
      title: 'Execution',
      items: [
        { name: 'Dashboard', path: '/dashboard' },
        { name: 'Marketing Planner', path: '/marketing-planner' },
        { name: 'Account Plan', path: '/account-plan' },
        { name: 'Insights Inbox', path: '/insights-inbox' },
      ],
    },
    {
      title: 'Library',
      items: [
        { name: 'Value Prop', path: '/value-prop' },
        { name: 'Personas', path: '/personas' },
        { name: 'ICP', path: '/icp' },
        { name: 'Arsenal', path: '/arsenal' },
      ],
    },
        {
          title: 'Geek Out',
          items: [
            { name: 'Models', path: '/models' },
            {name: 'Pains Map', path: '/pains-map' },
            { name: 'Thesis Builder', path: '/geek-out/thesis-builder' },
            { name: 'Simulator', path: '/reverse-case-studies' },
          ],
        },
    {
      title: 'Setup',
      items: [
        { name: 'CRM Setup', path: '/crm-setup' },
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
    const [arsenalMetaCount, setArsenalMetaCount] = useState(0);
    const [insightsPendingCount, setInsightsPendingCount] = useState(0);

    const refreshArsenalMetaCount = useCallback(async () => {
      if (typeof window === "undefined") return 0;
      const token = localStorage.getItem("token");
      if (!token) return 0;

      const headers = { Authorization: `Bearer ${token}` };
      const meRes = await fetch("http://localhost:8000/me", { headers });
      if (!meRes.ok) throw new Error("Failed to load user profile");
      const me = await meRes.json();

      const productsRes = await fetch(
        `http://localhost:8000/get-products/${me.company_id}`,
        { headers }
      );
      if (!productsRes.ok) throw new Error("Failed to load products");
      const productsData = await productsRes.json();
      const firstProductId = productsData.products?.[0]?.id;
      if (!firstProductId) return 0;

      const arsenalRes = await fetch(
        `http://localhost:8000/get-arsenal-library/${firstProductId}`,
        { headers }
      );
      if (!arsenalRes.ok) throw new Error("Failed to load arsenal");
      const arsenalData = await arsenalRes.json();
      const pendingAssets =
        (arsenalData?.arsenal?.assets || []).filter(
          (asset: any) => !asset.metadata_complete
        ).length || 0;
      const pendingChannels =
        (arsenalData?.arsenal?.channels || []).filter(
          (channel: any) => !channel.metadata_complete
        ).length || 0;

      return pendingAssets + pendingChannels;
    }, []);

    const refreshInsightsPendingCount = useCallback(async () => {
      if (typeof window === "undefined") return 0;
      const token = localStorage.getItem("token");
      if (!token) return 0;

      const headers = { Authorization: `Bearer ${token}` };
      const meRes = await fetch("http://localhost:8000/me", { headers });
      if (!meRes.ok) throw new Error("Failed to load profile");
      const me = await meRes.json();

      const productsRes = await fetch(
        `http://localhost:8000/get-products/${me.company_id}`,
        { headers }
      );
      if (!productsRes.ok) throw new Error("Failed to load products");
      const productsData = await productsRes.json();
      const firstProductId = productsData.products?.[0]?.id;
      if (!firstProductId) return 0;

      const insightsRes = await fetch(
        `http://localhost:8000/journey/global-thesis/${firstProductId}`,
        { headers }
      );
      if (!insightsRes.ok) throw new Error("Failed to load insights");
      const insightsData = await insightsRes.json();
      const insights = insightsData?.insights;
      if (!insights) return 0;
      return (
        (insights?.persona_recommendations?.length || 0) +
        (insights?.edge_recommendations?.length || 0)
      );
    }, []);

    useEffect(() => {
      let isMounted = true;
      refreshArsenalMetaCount()
        .then((count) => {
          if (isMounted) setArsenalMetaCount(count);
        })
        .catch(() => {
          if (isMounted) setArsenalMetaCount(0);
        });
      refreshInsightsPendingCount()
        .then((count) => {
          if (isMounted) setInsightsPendingCount(count);
        })
        .catch(() => {
          if (isMounted) setInsightsPendingCount(0);
        });
      return () => {
        isMounted = false;
      };
    }, [refreshArsenalMetaCount, refreshInsightsPendingCount]);

    useEffect(() => {
      if (
        location.pathname !== "/arsenal" &&
        location.pathname !== "/engagements-setup" &&
        location.pathname !== "/insights-inbox"
      ) {
        return;
      }
      let isMounted = true;
      if (location.pathname === "/arsenal" || location.pathname === "/engagements-setup") {
        refreshArsenalMetaCount()
          .then((count) => {
            if (isMounted) setArsenalMetaCount(count);
          })
          .catch(() => {
            if (isMounted) setArsenalMetaCount(0);
          });
      }
      if (location.pathname === "/insights-inbox") {
        refreshInsightsPendingCount()
          .then((count) => {
            if (isMounted) setInsightsPendingCount(count);
          })
          .catch(() => {
            if (isMounted) setInsightsPendingCount(0);
          });
      }
      return () => {
        isMounted = false;
      };
    }, [location.pathname, refreshArsenalMetaCount, refreshInsightsPendingCount]);

    useEffect(() => {
      const handler = (event: Event) => {
        const custom = event as CustomEvent<number>;
        if (typeof custom.detail === "number") {
          setInsightsPendingCount(custom.detail);
        }
      };
      window.addEventListener("insights:refresh-count", handler as EventListener);
      return () => {
        window.removeEventListener("insights:refresh-count", handler as EventListener);
      };
    }, []);
  
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
          className={`fixed lg:static top-0 left-0 h-full z-40 w-64 flex-shrink-0 bg-white px-4 py-6 space-y-8 overflow-y-auto transform transition-transform duration-200 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
        >
          <div className="flex items-center gap-2">
            <img
              src="/assets/getintentional-logo.png"
              alt="GetIntentional logo"
              className="max-h-10 w-auto mx-auto"
            />
          </div>
          <div className="mt-6 border-t pt-4">
            <SidebarActions className="" />
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
                    className={`flex items-center justify-between pl-4 pr-3 py-2 rounded-md text-sm font-medium transition ${
                      location.pathname === item.path
                        ? "bg-indigo-100 text-indigo-700"
                        : "hover:bg-gray-100"
                    }`}
                  >
                    <span>{item.name}</span>
                    {item.name === "Arsenal" && arsenalMetaCount > 0 && (
                      <span className="ml-3 inline-flex min-w-[1.75rem] justify-center rounded-full bg-red-500 px-2 py-0.5 text-xs font-semibold text-white">
                        {arsenalMetaCount > 99 ? "99+" : arsenalMetaCount}
                      </span>
                    )}
                    {item.name === "Insights Inbox" && insightsPendingCount > 0 && (
                      <span className="ml-3 inline-flex min-w-[1.75rem] justify-center rounded-full bg-blue-500 px-2 py-0.5 text-xs font-semibold text-white">
                        {insightsPendingCount > 99 ? "99+" : insightsPendingCount}
                      </span>
                    )}
                  </Link>
                ))}
              </nav>
            </div>
          ))}
        </aside>
  
        {/* Main content */}
        <main className="flex-1 min-w-0 p-4 lg:p-10 ml-0 transition-all duration-200 ease-in-out overflow-x-auto">
          <div className="max-w-[1800px] mx-auto w-full">
            {children}
          </div>
        </main>
      </div>
    );
  }
  
  export default AppLayout;
