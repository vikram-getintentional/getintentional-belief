import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import WebsiteAnalysis from './WebsiteAnalysis';
import SpinnerIcon from '../utils/SpinnerIcon';
import ErrorBoundary from "../components/ErrorBoundary";

const normalizeWebsiteInput = (input: string): { cleanDisplay: string; fullUrl: string } => {
  const withoutProtocol = input.replace(/^(https?:\/\/)?(www\.)?/, '');
  const cleanDisplay = withoutProtocol.replace(/\/$/, '');
  const fullUrl = `https://${cleanDisplay}`;
  return { cleanDisplay, fullUrl };
};

function ConfirmWebsite({ onAnalyzeSuccess }: { onAnalyzeSuccess?: (data: any) => void }) {
   const [email, setEmail] = useState('');
   const [websiteInput, setWebsiteInput] = useState('');
  const [cleanWebsite, setCleanWebsite] = useState('');
  const [fullWebsite, setFullWebsite] = useState('');
  const [confirmed, setConfirmed] = useState(false);

  const [scrapedText, setScrapedText] = useState('');
  const [plgCta, setPlgCta] = useState(false);
  const [footerFeatures, setFooterFeatures] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const loadingPhrases = [
    "Scanning homepage...",
    "Extracting value propositions...",
    "Identifying key capabilities...",
    "Structuring marketing insights...",
    "Finding product positioning...",
    "Reading solution pages...",
    "Learning how your company helps customers...",
  ];
  const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0);

  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/login');
      return;
    }

    fetch('http://localhost:8000/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.email) {
          setEmail(data.email);
          const domain = data.email.split('@')[1];
          const { cleanDisplay, fullUrl } = normalizeWebsiteInput(domain);
          setWebsiteInput(domain);
          setCleanWebsite(cleanDisplay);
          setFullWebsite(fullUrl);
        } else {
          navigate('/login');
        }
      });
  }, []);

  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setCurrentPhraseIndex((prev) => (prev + 1) % loadingPhrases.length);
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [loading]);

  const handleProceed = () => {
    const { cleanDisplay, fullUrl } = normalizeWebsiteInput(websiteInput);
    setCleanWebsite(cleanDisplay);
    setFullWebsite(fullUrl);
    setConfirmed(true);
  };

  useEffect(() => {
    if (confirmed && fullWebsite) {
      setLoading(true);
      setError('');
      const token = localStorage.getItem('token');

      fetch(`http://localhost:8000/scrape?url=${fullWebsite}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.error) {
            setError(data.error);
          } else {
            setScrapedText(data.text);
            setPlgCta(data.plg_cta_found);
            setFooterFeatures(data.footer_features || []);
          }
          setLoading(false);
        })
        .catch(() => {
          setError('Failed to scrape the website.');
          setLoading(false);
        });
    }
  }, [confirmed, fullWebsite]);

  return (
    <div className="space-y-2">
      {!confirmed ? (
        <>
          <div className="bg-white p-6 rounded-xl shadow-md space-y-4">
            <p className="text-sm text-gray-400 mb-1">
              Update your URL if this doesn't look right, or start analysis:
            </p>

            <div className="border-b border-gray-200 pb-1">
              <input
                type="text"
                value={websiteInput}
                onChange={(e) => setWebsiteInput(e.target.value)}
                className="bg-transparent text-xl font-semibold text-indigo-700 w-full focus:outline-none focus:ring-0 border-none p-0 m-0"
              />
            </div>
          </div>

          <div className="flex justify-end mt-4">
            <button
              disabled={loading}
              onClick={handleProceed}
              className={`px-4 py-2 rounded text-white flex items-center gap-2 ${
                loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {loading ? (
                <>
                  <SpinnerIcon />
                  Loading...
                </>
              ) : (
                <>Analyze</>
              )}
            </button>
          </div>
        </>
      ) : (
        <>
          <h3 className="text-lg font-semibold text-gray-800">
            Analyzing <span className="text-blue-600">{cleanWebsite}</span>...
          </h3>

          {loading && (
            <div className="text-blue-500 font-medium flex items-center gap-2 animate-pulse">
              <svg
                className="h-4 w-4 animate-spin text-blue-400"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z"
                />
              </svg>
              <span>{loadingPhrases[currentPhraseIndex]}</span>
            </div>
          )}

          {error && <p className="text-red-500">Error: {error}</p>}

          {!loading && !error && scrapedText && (
            <div className="mt-6 space-y-2">
              <ErrorBoundary>
                <WebsiteAnalysis
                  scrapedText={scrapedText}
                  url={fullWebsite}
                  plgCta={plgCta}
                  footerFeatures={footerFeatures}
                  onAnalyzeSuccess={onAnalyzeSuccess}
                />
              </ErrorBoundary>
              
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ConfirmWebsite;
