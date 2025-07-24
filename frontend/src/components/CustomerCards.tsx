import React from "react";

type Props = {
  profile: any; // Replace 'any' with a more specific type if you know the structure
};

const IdealCustomerCard = ({ profile }: Props) => (
  <div className="border rounded p-4 shadow bg-white">
    <h4 className="font-semibold mb-2">{profile.title || "Org Profile"}</h4>
    <pre className="text-xs text-gray-700">{JSON.stringify(profile, null, 2)}</pre>
  </div>
);

export default IdealCustomerCard;