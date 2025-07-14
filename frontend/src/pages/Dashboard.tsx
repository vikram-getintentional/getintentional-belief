import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
  const [user, setUser] = useState<any>(null);
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
      .then(res => res.json())
      .then(data => {
        if (data.detail) navigate('/login');
        else setUser(data);
      });
  }, []);

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold">Welcome to your Dashboard!</h1>
      {user && <p className="mt-2 text-gray-700">Logged in as: {user.email}</p>}

      <button
        onClick={() => {
          localStorage.removeItem('token');
          navigate('/login');
        }}
        className="mt-4 px-4 py-2 bg-red-500 text-white rounded"
      >
        Logout
      </button>
    </div>
  );
};

export default Dashboard;
