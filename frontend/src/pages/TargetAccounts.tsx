import React, { useEffect, useState } from "react";
import {
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper,
  TextField, Autocomplete, Chip, IconButton, CircularProgress,
  Button,
  MenuItem,
  Select
} from "@mui/material";
import SaveIcon from "@mui/icons-material/Save";
import DeleteIcon from "@mui/icons-material/Delete";

type Archetype = {
  label: string;
  industry?: string;
  revenue?: string;
  employees?: string;
  funding_stage?: string;
  geography?: string;
};

type TargetAccountRow = {
  id?: string;
  account_name: string;
  industry?: string;
  revenue?: string;
  employees?: string;
  funding_stage?: string;
  geography?: string;
  competitor_used: string[];
  other_tech_stack: string[];
  deal_status?: string;
};

function TargetAccounts() {
  const token = localStorage.getItem('token');
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [products, setProducts] = useState<any[]>([]);

  const [rows, setRows] = useState<TargetAccountRow[]>([]);
  const [loading, setLoading] = useState(false);

  // 1. Get company ID on mount
  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const meData = await meRes.json();
        setCompanyId(meData.company_id);

        // 2. Get product IDs for this company
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
        console.log("✅ Product IDs set:", prodData.products.map((p: any) => p.id));
        setProducts(prodData.products);

        // 3. If only one product, select it automatically
        if (prodData.products.length === 1) {
          console.log("✅ Automatically selecting single product:", prodData.products[0].id);
          setSelectedProductId(prodData.products[0].id);
        }
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    };
    fetchCompanyAndProducts();
  }, [token]);
  
  // Fetch entry parameters from graph - to be designed
  

  // Fetch existing target accounts
  useEffect(() => {
    if (!companyId || !selectedProductId) return;
    (async () => {
      setLoading(true);
      try {
        const res = await fetch(
          `http://localhost:8000/get-target-accounts/${companyId}?product_id=${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const data = await res.json();
        console.log("✅ Fetched Target Accounts:", data);
        // Map backend fields to frontend fields
        const mappedRows = (data.accounts || []).map((row: any) => ({
            ...row,
            revenue: row.revenue_range,
            employees: row.employee_range,
        }));
        setRows(mappedRows);
      } catch (e) {
        setRows([]);
      } finally {
        setLoading(false);
      }
    })();
  }, [companyId, selectedProductId, token]);

  // Only update row state on edit
  const handleRowChange = (idx: number, updated: Partial<TargetAccountRow>) => {
        setRows(rows.map((row, i) => i === idx ? { ...row, ...updated } : row));
    };

  // Save All
 const handleSaveAll = async () => {
  setLoading(true);
  for (const row of rows) {
    const rowToSave = {
      ...row,
      company_id: companyId,
      product_id: selectedProductId,
      revenue_range: row.revenue || null,
      employee_range: row.employees || null,
      funding_stage: row.funding_stage || null,
      geography: row.geography || null,
      competitor_used: row.competitor_used || [],
      other_tech_stack: row.other_tech_stack || [],
      deal_status: row.deal_status || "New"
    };
    console.log("Saving row:", rowToSave);
    await fetch(
      `http://localhost:8000/save-target-accounts/${companyId}?product_id=${selectedProductId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify(rowToSave)
      }
    );
  }
  setLoading(false);
};

  // Add empty row
  const handleAddRow = () => {
    setRows([
      ...rows,
      {
        account_name: "",
        industry: "",
        revenue: "",
        employees: "",
        funding_stage: "",
        geography: "",
        competitor_used: [],
        other_tech_stack: []
      }
    ]);
  };

  return (
    <TableContainer component={Paper} sx={{ mt: 2 }}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Target Account Name</TableCell>
            <TableCell>Industry</TableCell>
            <TableCell>Revenue Range</TableCell>
            <TableCell>Employee Range</TableCell>
            <TableCell>Funding Stage</TableCell>
            <TableCell>Geography</TableCell>
            <TableCell>Competitor Used</TableCell>
            <TableCell>Other Tech Stack</TableCell>
            <TableCell>Deal Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row, idx) => {
            return (
              <TableRow key={row.id || idx}>
                <TableCell>
                  <TextField
                    value={row.account_name}
                    onChange={e => handleRowChange(idx, { account_name: e.target.value })}
                    variant="standard"
                    placeholder="Enter account name"
                  />
                </TableCell>
                <TableCell>
                  <TextField
                    value={row.industry || ""}
                    onChange={e => handleRowChange(idx, { industry: e.target.value })}
                    variant="standard"
                    placeholder="SaaS, Ecom, Fintech.."
                  />
                </TableCell>
                <TableCell>
                  <Select
                    value={row.revenue || ""}
                    onChange={e => handleRowChange(idx, { revenue: e.target.value })}
                    variant="standard"
                    displayEmpty
                    fullWidth
                  >
                    <MenuItem value="">Select range…</MenuItem>
                    <MenuItem value="<$1M>">$0-1M</MenuItem>
                    <MenuItem value="$1-$10M">$1-10M</MenuItem>
                    <MenuItem value="$10M-$50M">$10-50M</MenuItem>
                    <MenuItem value="$50M-$200M">$50-200M</MenuItem>
                    <MenuItem value="$200M-$1B">$200-1B</MenuItem>
                    <MenuItem value=">$1B">$1B+</MenuItem>
                  </Select>
                </TableCell>

                <TableCell>
                  <Select
                    value={row.employees || ""}
                    onChange={e => handleRowChange(idx, { employees: e.target.value })}
                    variant="standard"
                    displayEmpty
                    fullWidth
                  >
                    <MenuItem value="">Select range…</MenuItem>
                    <MenuItem value="1-10">1-10</MenuItem>
                    <MenuItem value="11-50">11-50</MenuItem>
                    <MenuItem value="51-200">51-200</MenuItem>
                    <MenuItem value="201-1K">201-1K</MenuItem>
                    <MenuItem value="1K-5K">1K-5K</MenuItem>
                    <MenuItem value="5K-10K">5K-10K</MenuItem>
                    <MenuItem value=">10K">10K+</MenuItem>
                  </Select>
                </TableCell>

                <TableCell>
                  <Select
                    value={row.funding_stage || ""}
                    onChange={e => handleRowChange(idx, { funding_stage: e.target.value })}
                    variant="standard"
                    displayEmpty
                    fullWidth
                  >
                    <MenuItem value="">Select stage…</MenuItem>
                    <MenuItem value="Bootstrapped">Bootstrapped</MenuItem>
                    <MenuItem value="Pre-Seed">Pre-Seed</MenuItem>
                    <MenuItem value="Seed">Seed</MenuItem>
                    <MenuItem value="Series A">Series A</MenuItem>
                    <MenuItem value="Series B">Series B</MenuItem>
                    <MenuItem value="Series C+">Series C+</MenuItem>
                    <MenuItem value="Private">Private</MenuItem>
                    <MenuItem value="Public">Public</MenuItem>
                  </Select>
                </TableCell>

                <TableCell>
                  <Select
                    value={row.geography || ""}
                    onChange={e => handleRowChange(idx, { geography: e.target.value })}
                    variant="standard"
                    displayEmpty
                    fullWidth
                  >
                    <MenuItem value="">Select region…</MenuItem>
                    <MenuItem value="North America">North America</MenuItem>
                    <MenuItem value="Latin America">Latin America</MenuItem>
                    <MenuItem value="EMEA">EMEA</MenuItem>
                    <MenuItem value="APAC">APAC</MenuItem>
                    <MenuItem value="ANZ">ANZ</MenuItem>
                    <MenuItem value="Global">Global</MenuItem>
                  </Select>
                </TableCell>
                <TableCell>
                  <Autocomplete
                    multiple
                    freeSolo
                    options={[]}
                    value={row.competitor_used}
                    onChange={(_, newValue) => handleRowChange(idx, { competitor_used: newValue })}
                    renderTags={(value, getTagProps) =>
                        value.map((option, index) => {
                            const tagProps = getTagProps({ index });
                            const { key, ...rest } = tagProps;
                            return (
                                <Chip
                                    variant="outlined"
                                    label={option}
                                    key={key}
                                    {...rest}
                                />
                            );
                        })
                    }
                    renderInput={params => <TextField {...params} variant="standard" placeholder="Add competitors" />}
                  />
                </TableCell>
                <TableCell>
                  <Autocomplete
                    multiple
                    freeSolo
                    options={[]}
                    value={row.other_tech_stack}
                    onChange={(_, newValue) => handleRowChange(idx, { other_tech_stack: newValue })}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => {
                        const tagProps = getTagProps({ index });
                        const { key, ...rest } = tagProps;
                        return (
                          <Chip
                            variant="outlined"
                            label={option}
                            key={key}
                            {...rest}
                          />
                        );
                      })
                    }
                    renderInput={params => <TextField {...params} variant="standard" placeholder="Add tech stack" />}
                  />
                </TableCell>
                
                <TableCell>
                  <Select
                    value={row.deal_status || ""}
                    onChange={e => handleRowChange(idx, { deal_status: e.target.value })}
                    variant="standard"
                    displayEmpty
                    fullWidth
                  >
                    <MenuItem value="">Select status…</MenuItem>
                    <MenuItem value="New">New</MenuItem>
                    <MenuItem value="In-Progress">In-Progress</MenuItem>
                    <MenuItem value="Closed-Won">Closed-Won</MenuItem>
                    <MenuItem value="Closed-Lost">Closed-Lost</MenuItem>
                  </Select>
                </TableCell>
                <TableCell>
                    <IconButton
                        color="error"
                        onClick={async () => {
                        if (!companyId || !selectedProductId || !row.id) return;
                        await fetch(
                            `http://localhost:8000/delete-target-account/${companyId}/${row.id}?product_id=${selectedProductId}`,
                            {
                            method: "DELETE",
                            headers: { Authorization: `Bearer ${token}` }
                            }
                        );
                        // Remove from local state
                        setRows(rows.filter((r, i) => i !== idx));
                        }}
                    >
                        <DeleteIcon />
                    </IconButton>
                </TableCell>
              </TableRow>
            );
          })}
          <TableRow>
            <TableCell colSpan={9}>
              <IconButton onClick={handleAddRow} color="primary">
                <SaveIcon /> Add Row
              </IconButton>
              {loading && <CircularProgress size={20} sx={{ ml: 2 }} />}
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
      <Button
        variant="contained"
        color="primary"
        startIcon={<SaveIcon />}
        onClick={handleSaveAll}
        sx={{ mt: 2 }}
        >
        Save All
    </Button>
    </TableContainer>
    
  );
}

export default TargetAccounts;
