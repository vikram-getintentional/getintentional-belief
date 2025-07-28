export const getCardClass = (relevance: number) => {
  if (relevance > 0.8) {
    return "mb-4 p-4 ring-2 ring-indigo-500 bg-indigo-50 rounded";
  }
  if (relevance > 0.6) {
    return "mb-4 p-4 ring-2 ring-indigo-200 bg-indigo-50 rounded";
  }
   else if (relevance > 0.4) {
    return "mb-4 p-4 ring-1 ring-gray-200 bg-white rounded";
  } else {
    return "mb-4 p-4 ring-1 ring-gray-100 bg-white rounded";
  }
};