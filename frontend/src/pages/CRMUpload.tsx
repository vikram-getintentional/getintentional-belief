import React, { useEffect, useState } from 'react';
import {
  Container, Typography, TextField, Button, Paper, IconButton, MenuItem, List, ListItem, ListItemText
} from '@mui/material';
import Grid from '@mui/material/Grid';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';

const industries = ['Technology', 'Finance', 'Healthcare', 'Retail'];
const revenueRanges = ['<$10M', '$10M-$50M', '$50M-$200M', '>$200M'];
const employeeRanges = ['<50', '50-200', '200-1000', '>1000'];
const fundingStages = ['Seed', 'Series A', 'Series B', 'Series C', 'Public'];
const geographies = ['North America', 'Europe', 'Asia', 'Other'];
const departments = ['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'];
const seniorities = ['Entry', 'Mid', 'Senior', 'Director', 'VP', 'C-Level'];

type Persona = {
  personaName: string;
  role: string;
  department: string;
  seniority: string;
  jobDescription: string[];
};

type Account = {
  accountName: string;
  industry: string;
  revenueRange: string;
  employeeRange: string;
  fundingStage: string;
  geography: string;
  personas: Persona[];
};

function CRMUpload() {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [accounts, setAccounts] = useState<Account[]>([
    {
      accountName: '',
      industry: '',
      revenueRange: '',
      employeeRange: '',
      fundingStage: '',
      geography: '',
      personas: [
        {
          personaName: '',
          role: '',
          department: '',
          seniority: '',
          jobDescription: [''],
        },
      ],
    },
  ]);

  const [statusMsg, setStatusMsg] = useState('');
  const token = localStorage.getItem('token');

  // Fetch company and products on mount
  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch('http://localhost:8000/me', {
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
          setStatusMsg('No products found. Please run Value Prop first.');
          return;
        }
        setProducts(prodData.products);
        if (prodData.products.length === 1) {
          setSelectedProductId(prodData.products[0].id);
        }
      } catch (err) {
        setStatusMsg('Error fetching company or products.');
      }
    };
    fetchCompanyAndProducts();
  }, [token]);
  

  const handleAccountChange = (idx: number, field: keyof Account, value: string) => {
    const updated = [...accounts];
    if (field !== 'personas') {
      updated[idx][field] = value as any;
    }
    setAccounts(updated);
  };

  const handlePersonaChange = (
    accIdx: number,
    personaIdx: number,
    field: keyof Persona,
    value: string | string[]
  ) => {
    const updated = [...accounts];
    updated[accIdx].personas[personaIdx][field] = value as any;
    setAccounts(updated);
  };

  const handleJobDescChange = (accIdx: number, personaIdx: number, descIdx: number, value: string) => {
    const updated = [...accounts];
    updated[accIdx].personas[personaIdx].jobDescription[descIdx] = value;
    setAccounts(updated);
  };

  const addAccount = () => {
    setAccounts([
      ...accounts,
      {
        accountName: '',
        industry: '',
        revenueRange: '',
        employeeRange: '',
        fundingStage: '',
        geography: '',
        personas: [
          {
            personaName: '',
            role: '',
            department: '',
            seniority: '',
            jobDescription: [''],
          },
        ],
      },
    ]);
  };

  const removeAccount = (idx) => {
    setAccounts(accounts.filter((_, i) => i !== idx));
  };

  const addPersona = (accIdx) => {
    const updated = [...accounts];
    updated[accIdx].personas.push({
      personaName: '',
      role: '',
      department: '',
      seniority: '',
      jobDescription: [''],
    });
    setAccounts(updated);
  };

  const removePersona = (accIdx: number, personaIdx: number) => {
    const updated = [...accounts];
    updated[accIdx].personas = updated[accIdx].personas.filter((_, i) => i !== personaIdx);
    setAccounts(updated);
  };

  const addJobDesc = (accIdx: number, personaIdx: number) => {
    const updated = [...accounts];
    updated[accIdx].personas[personaIdx].jobDescription.push('');
    setAccounts(updated);
  };

  const removeJobDesc = (accIdx: number, personaIdx: number, descIdx: number) => {
    const updated = [...accounts];
    updated[accIdx].personas[personaIdx].jobDescription = updated[accIdx].personas[personaIdx].jobDescription.filter((_, i) => i !== descIdx);
    setAccounts(updated);
  };

  // Save handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!companyId || !selectedProductId) {
      setStatusMsg('Please select a product.');
      return;
    }
    try {
      const response = await fetch(
        `http://localhost:8000/save/save-target-account-list/${companyId}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            product_id: selectedProductId,
            accounts,
          }),
        }
      );
      if (!response.ok) {
        throw new Error('Failed to save accounts');
      }
      setStatusMsg('Accounts saved successfully!');
    } catch (error) {
      setStatusMsg('Error saving accounts.');
      console.error(error);
    }
  };


  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>Upload Target Accounts</Typography>
      <form onSubmit={handleSubmit}>
        {accounts.map((account, accIdx) => (
          <Paper key={accIdx} sx={{ p: 2, mb: 3 }}>
            <Grid container spacing={2}>
              {/* Account Name on its own line */}
              <Grid item size={{ xs: 12 }}>
                <TextField
                  label="Account Name"
                  fullWidth
                  value={account.accountName}
                  onChange={e => handleAccountChange(accIdx, 'accountName', e.target.value)}
                />
              </Grid>
              {/* All other fields grouped on the next line */}
              <Grid item size={{ xs: 12 }}>
                <Grid container spacing={2}>
                  <Grid item size={{ xs: 4, sm: 4 }}>
                    <TextField
                      select
                      label="Industry"
                      fullWidth
                      value={account.industry}
                      onChange={e => handleAccountChange(accIdx, 'industry', e.target.value)}
                    >
                      {industries.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </TextField>
                  </Grid>
                  <Grid item size={{ xs: 4, sm: 4 }}>
                    <TextField
                      select
                      label="Revenue Range"
                      fullWidth
                      value={account.revenueRange}
                      onChange={e => handleAccountChange(accIdx, 'revenueRange', e.target.value)}
                    >
                      {revenueRanges.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </TextField>
                  </Grid>
                  <Grid item size={{ xs: 6, sm: 4 }}>
                    <TextField
                      select
                      label="Employee Range"
                      fullWidth
                      value={account.employeeRange}
                      onChange={e => handleAccountChange(accIdx, 'employeeRange', e.target.value)}
                    >
                      {employeeRanges.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </TextField>
                  </Grid>
                  <Grid item size={{ xs: 6, sm: 4 }}>
                    <TextField
                      select
                      label="Funding Stage"
                      fullWidth
                      value={account.fundingStage}
                      onChange={e => handleAccountChange(accIdx, 'fundingStage', e.target.value)}
                    >
                      {fundingStages.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </TextField>
                  </Grid>
                  <Grid item size={{ xs: 6, sm: 4 }}>
                    <TextField
                      select
                      label="Geography"
                      fullWidth
                      value={account.geography}
                      onChange={e => handleAccountChange(accIdx, 'geography', e.target.value)}
                    >
                      {geographies.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                    </TextField>
                  </Grid>
                </Grid>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6">Identified Personas</Typography>
                {account.personas.map((persona, personaIdx) => (
                  <Paper key={personaIdx} sx={{ p: 2, mb: 2 }}>
                    <Grid container spacing={2}>
                      <Grid item size={{ xs: 12, sm: 12 }}>
                        <TextField
                          label="Persona Name"
                          fullWidth
                          value={persona.personaName}
                          onChange={e => handlePersonaChange(accIdx, personaIdx, 'personaName', e.target.value)}
                        />
                      </Grid>
                      <Grid item size={{ xs: 6, sm: 4 }}>
                        <TextField
                          label="Role"
                          fullWidth
                          value={persona.role}
                          onChange={e => handlePersonaChange(accIdx, personaIdx, 'role', e.target.value)}
                        />
                      </Grid>
                      <Grid item size={{ xs: 6, sm: 4 }}>
                        <TextField
                          select
                          label="Department"
                          fullWidth
                          value={persona.department}
                          onChange={e => handlePersonaChange(accIdx, personaIdx, 'department', e.target.value)}
                        >
                          {departments.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                        </TextField>
                      </Grid>
                      <Grid item size={{ xs: 6, sm: 4 }}>
                        <TextField
                          select
                          label="Seniority"
                          fullWidth
                          value={persona.seniority}
                          onChange={e => handlePersonaChange(accIdx, personaIdx, 'seniority', e.target.value)}
                        >
                          {seniorities.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                        </TextField>
                      </Grid>
                      <Grid item size={{ xs: 12, sm: 10 }}>
                        <Typography variant="subtitle1">Job Description (bulleted)</Typography>
                        <List>
                          {persona.jobDescription.map((desc, descIdx) => (
                            <ListItem key={descIdx} disablePadding>
                              <TextField
                                fullWidth
                                placeholder={`Bullet ${descIdx + 1}`}
                                value={desc}
                                onChange={e => handleJobDescChange(accIdx, personaIdx, descIdx, e.target.value)}
                                sx={{ mr: 1 }}
                              />
                              <IconButton onClick={() => removeJobDesc(accIdx, personaIdx, descIdx)} disabled={persona.jobDescription.length === 1}>
                                <DeleteIcon />
                              </IconButton>
                            </ListItem>
                          ))}
                        </List>
                        <Button size="small" onClick={() => addJobDesc(accIdx, personaIdx)} startIcon={<AddIcon />}>
                          Add Bullet
                        </Button>
                      </Grid>
                      <Grid item xs={12}>
                        <Button color="error" size="small" onClick={() => removePersona(accIdx, personaIdx)} startIcon={<DeleteIcon />} disabled={account.personas.length === 1}>
                          Remove Persona
                        </Button>
                      </Grid>
                    </Grid>
                  </Paper>
                ))}
                <Button size="small" onClick={() => addPersona(accIdx)} startIcon={<AddIcon />}>
                  Add Persona
                </Button>
              </Grid>
              <Grid item xs={12}>
                <Button color="error" size="small" onClick={() => removeAccount(accIdx)} startIcon={<DeleteIcon />} disabled={accounts.length === 1}>
                  Remove Account
                </Button>
              </Grid>
            </Grid>
          </Paper>
        ))}
        <Button variant="contained" onClick={addAccount} startIcon={<AddIcon />} sx={{ mr: 2 }}>
          Add Account
        </Button>
        <Button variant="contained" color="primary" type="submit">
          Save
        </Button>
      </form>
    </Container>
  );
}

export default CRMUpload;