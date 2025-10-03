
import { useEffect, useState } from "react";
import { Container, Tabs, Tab, Box, Typography } from "@mui/material";
import AssetsTemplate from "../components/AssetsTemplate";

const AdminPane = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const token = localStorage.getItem("token");
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
    const fetchAdmin = async () => {
        console.log("Fetching admin data for product:", selectedProductId);
    };
    fetchAdmin();
  }, [selectedProductId]);

  return (
    <Container maxWidth="lg" sx={{ mt: 0 }}>
      <Typography variant="h4" gutterBottom>Admin & Setup</Typography>
      <Tabs
        value={tabIndex}
        onChange={(_, newValue) => setTabIndex(newValue)}
        indicatorColor="primary"
        textColor="primary"
        variant="fullWidth"
        sx={{ mb: 3 }}
      >
        <Tab label="Assets Template" />
        <Tab label="Campaigns Template" />
        <Tab label="CRM Setup" />
        

      </Tabs>

      <Box hidden={tabIndex !== 0}>
        <Typography variant="h6" gutterBottom>Asset Setup</Typography>
        {/* Asset setup content goes here */}
        <AssetsTemplate />

      </Box>
      <Box hidden={tabIndex !== 1}>
        <Typography variant="h6" gutterBottom>Campaigns Setup</Typography>
        {/* Campaigns setup content goes here */}
      </Box>
      <Box hidden={tabIndex !== 2}>
        <Typography variant="h6" gutterBottom>CRM Setup</Typography>
        {/* CRM setup content goes here */}
      </Box>
    </Container>
  );
};

export default AdminPane;