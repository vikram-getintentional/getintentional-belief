import { useEffect, useState } from "react";
import PersonaCard from "../components/PersonaCard";

const Personas = () => {
  const [personas, setPersonas] = useState([]);
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchPersonas = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const meData = await meRes.json();
        const companyId = meData.company_id;

        const personaRes = await fetch(
          `http://localhost:8000/company-personas/${companyId}`
        );
        const personaData = await personaRes.json();
        setPersonas(personaData);
      } catch (err) {
        console.error("Error fetching personas:", err);
      }
    };

    fetchPersonas();
  }, []);

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold mb-6">Saved Personas</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {personas.map((p, i) => (
          <PersonaCard
            key={i}
            persona={p}
            selectedPains={p.pains || []}
            isSelected={true}
            onTogglePain={() => {}}
            onTogglePersona={() => {}}
          />
        ))}
      </div>
    </div>
  );
};

export default Personas;
