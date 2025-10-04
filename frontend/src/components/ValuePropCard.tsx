import React from "react";

type ValuePropCardProps = {
  product_node_id: string;
  summary: string;
  plg_flag?: boolean;
};

const ValuePropCard: React.FC<ValuePropCardProps> = ({
  product_node_id,
  summary,
  plg_flag,
}) => (
  <div className="bg-white rounded-lg shadow-md p-6 border border-gray-100 flex flex-col h-full">
    <div className="mb-2 flex items-center justify-between">
      <span className="text-xs text-gray-400">Product ID: {product_node_id}</span>
      {plg_flag && (
        <span className="bg-green-100 text-green-700 text-xs px-2 py-1 rounded font-semibold ml-2">
          PLG
        </span>
      )}
    </div>
    <div>
      <h3 className="text-lg font-bold text-indigo-700 mb-1">Product Summary</h3>
      <p className="text-gray-700 text-sm">{summary}</p>
    </div>
  </div>
);

export default ValuePropCard;