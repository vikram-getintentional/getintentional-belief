let initialized = false;

const redirectToLogin = () => {
  try {
    localStorage.removeItem('token');
  } catch (err) {
    // ignore if storage is unavailable
  }
  if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
    window.location.replace('/login');
  }
};

export function initUnauthorizedRedirect() {
  if (typeof window === 'undefined' || initialized) {
    return;
  }
  initialized = true;

  const originalFetch = window.fetch.bind(window);

  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    if (response.status === 401) {
      redirectToLogin();
      throw new Error('Unauthorized');
    }
    return response;
  };
}
