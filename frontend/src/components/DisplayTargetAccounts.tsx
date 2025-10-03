import React, { useEffect, useState } from "react";
import { Container, Typography, Button, Paper, List, ListItem, ListItemText } from "@mui/material";

type Account = {
  accountName: string;
  industry: string;
  revenueRange: string;
};

interface DisplayTargetAccountsProps {
  companyId: string;
  productId: string;
  onAddNew: () => void;
}

const DisplayTargetAccounts: React.FC<DisplayTargetAccountsProps> = ({ companyId, productId, onAddNew }) => {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [loading, setLoading] = useState(true);
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchAccounts = async () => {
      const res = await fetch(`http://localhost:8000/get-target-accounts/${companyId}?product_id=${productId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      setAccounts(data.accounts ?? []);
      setLoading(false);
    };
    fetchAccounts();
  }, [companyId, productId, token]);

  if (loading) return <Typography>Loading...</Typography>;

  return (
    <Container>
      <Typography variant="h5" gutterBottom>Target Accounts</Typography>
      <List>
        {accounts.map((account, idx) => (
          <Paper key={idx} sx={{ mb: 2, p: 2 }}>
            <ListItem>
              <ListItemText
                primary={account.accountName}
                secondary={`Industry: ${account.industry}, Revenue: ${account.revenueRange}`}
              />
            </ListItem>
          </Paper>
        ))}
      </List>
      <Button variant="contained" onClick={onAddNew}>Add New Target Account</Button>
    </Container>
  );
};

export default DisplayTargetAccounts;