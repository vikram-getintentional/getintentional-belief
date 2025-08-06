import { getCardClass } from "./interfaceElements/cardUtils";
import Pill from "./interfaceElements/pillbox";

type IcpCardProps = {
  icp: {
    industry: string;
    revenue: string;
    employees: string;
    funding_stage: string;
    geography: string;
    relevance: number;
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
  <div className={getCardClass(icp.relevance)}>
    <h3 className="font-bold text-lg mb-2">ICP Profile</h3>
    <div className="mb-2">
      <span className={getPillClass(icp.relevance)}>
        Relevance: {(icp.relevance * 100).toFixed(1)}%
      </span>
    </div>
    <ul className="mb-2 space-y-2">
      <li>
        <b>Industry:</b>{" "}
        <span className="inline-block bg-gray-100 text-gray-800 border border-gray-300 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full">
          {icp.industry}
        </span>
      </li>
      <li>
        <b>Revenue:</b>{" "}
        <span className="inline-block bg-gray-100 text-gray-800 border border-gray-300 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full">
          {icp.revenue}
        </span>
      </li>
      <li>
        <b>Employees:</b>{" "}
        <span className="inline-block bg-gray-100 text-gray-800 border border-gray-300 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full">
          {icp.employees}
        </span>
      </li>
      <li>
        <b>Funding Stage:</b>{" "}
        <span className="inline-block bg-gray-100 text-gray-800 border border-gray-300 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full">
          {icp.funding_stage}
        </span>
      </li>
      <li>
        <b>Geography:</b>{" "}
        <span className="inline-block bg-gray-100 text-gray-800 border border-gray-300 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-full">
          {icp.geography}
        </span>
      </li>
    </ul>
  </div>
);

export default IcpCard;