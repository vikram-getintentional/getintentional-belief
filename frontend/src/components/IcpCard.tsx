type IcpCardProps = {
  icp: {
    industries: [string, number][];
    revenues: [string, number][];
    employees: [string, number][];
    funding_stages: [string, number][];
    geographies: [string, number][];
  };
};

// Utility to pick pill color based on relevance
const getPillClass = (score: number) => {
  if (score > 0.8) {
    // Dark blue, white text, dark border
    return "inline-block bg-blue-500 text-white border border-blue-600 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full";
  } else if (score > 0.5) {
    // Medium blue, white text, dark border
    return "inline-block bg-blue-200 text-white border border-blue-400 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full";
  } else {
    // Very light blue, blue text, blue border
    return "inline-block bg-blue-100 text-blue-300 border border-blue-300 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full";
  }
};

const IcpCard = ({ icp }: IcpCardProps) => (
  <div className="mb-8 p-4 border rounded bg-white shadow">
    <h3 className="font-bold text-lg mb-2">ICP Profile</h3>
    <ul className="mb-2 space-y-2">
      <li>
        <b>Industries:</b>{" "}
        {icp.industries && icp.industries.length > 0 ? (
          icp.industries.map(([label, score], idx) => (
            <span key={idx} className={getPillClass(score)}>
              {label}
            </span>
          ))
        ) : (
          <span className="text-gray-400">None</span>
        )}
      </li>
      <li>
        <b>Revenues:</b>{" "}
        {icp.revenues && icp.revenues.length > 0 ? (
          icp.revenues.map(([label, score], idx) => (
            <span key={idx} className={getPillClass(score)}>
              {label}
            </span>
          ))
        ) : (
          <span className="text-gray-400">None</span>
        )}
      </li>
      <li>
        <b>Employees:</b>{" "}
        {icp.employees && icp.employees.length > 0 ? (
          icp.employees.map(([label, score], idx) => (
            <span key={idx} className={getPillClass(score)}>
              {label}
            </span>
          ))
        ) : (
          <span className="text-gray-400">None</span>
        )}
      </li>
      <li>
        <b>Funding Stages:</b>{" "}
        {icp.funding_stages && icp.funding_stages.length > 0 ? (
          icp.funding_stages.map(([label, score], idx) => (
            <span key={idx} className={getPillClass(score)}>
              {label}
            </span>
          ))
        ) : (
          <span className="text-gray-400">None</span>
        )}
      </li>
      <li>
        <b>Geographies:</b>{" "}
        {icp.geographies && icp.geographies.length > 0 ? (
          icp.geographies.map(([label, score], idx) => (
            <span key={idx} className={getPillClass(score)}>
              {label}
            </span>
          ))
        ) : (
          <span className="text-gray-400">None</span>
        )}
      </li>
    </ul>
  </div>
);

export default IcpCard;