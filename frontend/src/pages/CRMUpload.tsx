import React, { useState } from "react";



function CRMUpload() {
  const token = localStorage.getItem('token');

  const [companyId, setCompanyId] = useState<string | null>(null);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  
  const [statusMsg, setStatusMsg] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);
  const [showUpload, setShowUpload] = useState(false);

  const handleAddNew = () => setShowUpload(true);
   const handleSave = () => {
    setShowUpload(false);
    setRefreshKey(k => k + 1); // trigger refresh
  };
  const handleCancel = () => setShowUpload(false);

  return (
    <div>CRM Upload Page - Coming Soon!</div>
  );
}

export default CRMUpload;