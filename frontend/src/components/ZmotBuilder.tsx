import { useState, useEffect } from "react";

type ZmotBuilderProps = {
  productId: string;
  onSave: () => void;
};

const ZmotBuilder: React.FC<ZmotBuilderProps> = ({ productId, onSave }) => {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState<any>(null);

  const handleInferClick = async () => {
    setLoading(true);
    setOutput(null);
    try {
      const res = await fetch(
        `http://localhost:8000/analyze/generate_zmot_icp/${productId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("token") || ""}`,
          }
        }
      );
      const data = await res.json();
      setOutput(data);
      if (onSave) onSave();
    } catch (err) {
      setOutput({ error: "Failed to fetch ZMOT & ICP data." });
    } finally {
      setLoading(false);
    }
  };

  // Automatically trigger inference on mount
  useEffect(() => {
    handleInferClick();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productId]);

  return (
    <div>
      {loading && <div>Loading...</div>}
      {output && <pre>{JSON.stringify(output, null, 2)}</pre>}
    </div>
  );
};

export default ZmotBuilder;