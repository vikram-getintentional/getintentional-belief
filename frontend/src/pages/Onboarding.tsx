import ConfirmWebsite from '../components/ConfirmWebsite';
import { useEffect, useState } from 'react';

// const getCompanyNameFromEmail = (email: string) => {
//   const domain = email.split('@')[1]?.split('.')[0]; // acme.com → acme
//   return domain ? domain.charAt(0).toUpperCase() + domain.slice(1) : '';
// };


const Onboarding = () => {
  const [_, setEmail] = useState('');
  const [companyName, setCompanyName] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) return;

    fetch('http://localhost:8000/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.email) {
          setEmail(data.email);
          setCompanyName(data.company_name);
        }
      });
  }, []);
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-8">
       
       <div className="w-full max-w-7xl mx-auto px-10 py-10 space-y-8">
       <h1 className="text-4xl font-bold text-gray-900 mb-6">
          {companyName}&#39;s Marketing Thesis
        </h1>
        <ConfirmWebsite />
        {/* 🔮 Future: Add value prop extraction results below */}
      </div>
    </div>
  );
};

export default Onboarding;
