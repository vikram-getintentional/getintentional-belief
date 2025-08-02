type me = { companyID: string };

async function api(endpoint: string, token: string): Promise<any> {
  const headers = {
    Authorization: `Bearer ${token}`,
  };
  const response = await fetch(`/api${endpoint}`, { headers });
  if (response.status === 200) {
    return response.json();
  } else {
    // TODO handle invalid token, network failures separately
    throw new Error("Unable to fetch session");
  }
}

export async function me(token: string): Promise<me> {
  const json = await api("/me", token);
  return { companyID: json.company_id };
}

type product = {
  id: string;
  name: string;
};

export async function products(
  token: string,
  companyID: string,
): Promise<{ products: product[] }> {
  const json = await api(`/get-products/${companyID}`, token);
  return { products: json.products || [] }; // TODO API errors??
}

export async function personas(
  token: string,
  productID: string,
): Promise<any[]> {
  const json = await api(`/get-personas/${productID}`, token);
  if (json.details) {
    throw new Error(json.details);
  } else {
    return json;
  }
}
