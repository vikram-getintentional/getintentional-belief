import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
  e.preventDefault();
  setError('');

  try {
    const res = await fetch('http://localhost:8000/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
      throw new Error('Invalid email or password');
    }

    const data = await res.json();
    localStorage.setItem('token', data.access_token);

    // Assume company_id is returned in login response
    const companyId = data.company_id;

    // Check if product subgraph exists for this company
    const subgraphRes = await fetch(`http://localhost:8000/company/${companyId}/product-subgraph`, {
      headers: {
        'Authorization': `Bearer ${data.access_token}`,
        'Content-Type': 'application/json',
      },
    });

    if (!subgraphRes.ok) {
      throw new Error('Failed to check product subgraph');
    }

    const subgraphData = await subgraphRes.json();
    // subgraphData should have { exists: boolean, capabilities: [...] }
    if (subgraphData.exists && Array.isArray(subgraphData.capabilities) && subgraphData.capabilities.length > 0) {
      navigate('/valueproposition');
    } else {
      navigate('/onboarding');
    }
  } catch (err: any) {
    setError(err.message || 'Login failed');
  }
};

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <form
        onSubmit={handleLogin}
        className="bg-white p-6 rounded-xl shadow-md space-y-4 w-80"
      >
        <h2 className="text-2xl font-bold text-center">Login</h2>

        {error && (
          <div className="text-red-600 text-sm text-center">{error}</div>
        )}

        <input
          type="email"
          placeholder="Email"
          className="w-full border border-gray-300 p-2 rounded"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full border border-gray-300 p-2 rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        <button
          type="submit"
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 transition"
        >
          Login
        </button>
      </form>
    </div>
  );
};

export default Login;
