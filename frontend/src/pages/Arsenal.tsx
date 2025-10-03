import { useEffect, useState } from "react";
import {
  Container, Typography, Select, MenuItem, FormControl, InputLabel, Paper, Box, Grid, Tabs, Tab
} from "@mui/material";
import AssetsLibrary from "../components/AssetsLibrary";
import ChannelsLibrary from "../components/ChannelsLibrary";

const Arsenal = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const token = localStorage.getItem("token");
  const [arsenalAssets, setArsenalAssets] = useState<any[]>([]);
  const [arsenalChannels, setArsenalChannels] = useState<any[]>([]);
  const [tabIndex, setTabIndex] = useState(0);

  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const meData = await meRes.json();
        setCompanyId(meData.company_id);

        const prodRes = await fetch(
          `http://localhost:8000/get-products/${meData.company_id}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const prodData = await prodRes.json();
        if (!prodData.products || prodData.products.length === 0) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        setProducts(prodData.products);
        if (prodData.products.length === 1) {
          setSelectedProductId(prodData.products[0].id);
        }
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    };
    fetchCompanyAndProducts();
  }, [token]);

  useEffect(() => {
    if (!selectedProductId) return;
    const fetchArsenal = async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/get-arsenal-library/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const data = await res.json();
        setArsenalAssets(data.arsenal.assets ?? []);
        setArsenalChannels(data.arsenal.channels ?? []);
        console.log("✅ Fetched Arsenal Library:", data);
      } catch (err) {
        setStatusMsg("Error fetching Arsenal Library.");
      }
    };
    fetchArsenal();
  }, [selectedProductId]);

  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>Arsenal Library</Typography>
      {products.length > 1 && (
        <Box mb={3}>
          <FormControl fullWidth>
            <InputLabel>Select Product</InputLabel>
            <Select
              value={selectedProductId || ""}
              label="Select Product"
              onChange={e => setSelectedProductId(e.target.value)}
            >
              <MenuItem value="" disabled>Select a product</MenuItem>
              {products.map(p => (
                <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      )}
      {statusMsg && <Typography color="error" mb={2}>{statusMsg}</Typography>}

      <Tabs
        value={tabIndex}
        onChange={(_, newValue) => setTabIndex(newValue)}
        indicatorColor="primary"
        textColor="primary"
        variant="fullWidth"
        sx={{ mb: 3 }}
      >
        <Tab label="Assets" />
        <Tab label="Channels" />
        <Tab label="Campaigns" />
      </Tabs>
      <Box hidden={tabIndex !== 0}>
        <Typography variant="h5" gutterBottom>Assets</Typography>
        <Grid container spacing={2} mb={4} alignItems="stretch" whiteSpace={"normal"}>
          <AssetsLibrary assets={arsenalAssets} />
        </Grid>
      </Box>
      <Box hidden={tabIndex !== 1}>
        <Typography variant="h5" gutterBottom>Channels</Typography>
        <Grid container spacing={2}>
          <ChannelsLibrary channels={arsenalChannels} />
        </Grid>
      </Box>
    </Container>
  );
};

export default Arsenal;