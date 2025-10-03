import React, { useState } from "react";
import { Container, Typography, TextField, Button, Paper, MenuItem } from "@mui/material";

const industries = ['Technology', 'Finance', 'Healthcare', 'Retail'];

const UploadTargetAccount = ({ companyId, productId, onSave, onCancel }) => {
  const [account, setAccount] = useState({
    accountName: "",
    industry: "",
    // ...other fields...
  });
  const token = localStorage.getItem("token");

  const handleChange = (field, value) => {
    setAccount({ ...account, [field]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await fetch(`http://localhost:8000/save/save-target-account-list/${companyId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        product_id: productId,
        accounts: [account],
      }),
    });
    onSave();
  };

  return (
    <Container>
      <Typography variant="h5" gutterBottom>Add Target Account</Typography>
      <form onSubmit={handleSubmit}>
        <Paper sx={{ p: 2, mb: 2 }}>
          <TextField
            label="Account Name"
            fullWidth
            value={account.accountName}
            onChange={e => handleChange("accountName", e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            select
            label="Industry"
            fullWidth
            value={account.industry}
            onChange={e => handleChange("industry", e.target.value)}
            sx={{ mb: 2 }}
          >
            {industries.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
          </TextField>
          {/* Add other fields as needed */}
        </Paper>
        <Button variant="contained" type="submit" sx={{ mr: 2 }}>Save</Button>
        <Button variant="outlined" onClick={onCancel}>Cancel</Button>
      </form>
    </Container>
  );
};

export default UploadTargetAccount;