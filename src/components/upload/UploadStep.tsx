import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface UploadStepProps {
  icon: LucideIcon;
  title: string;
  description: string;
  status: "pending" | "active" | "complete" | "error" | "warning";
}

const statusColors = {
  pending: "bg-muted text-muted-foreground",
  active: "gradient-primary text-primary-foreground animate-pulse",
  complete: "bg-verified text-verified-foreground",
  error: "bg-destructive text-destructive-foreground",
  warning: "bg-warning text-warning-foreground",
};

const statusBorderColors = {
  pending: "border-muted",
  active: "border-primary",
  complete: "border-verified",
  error: "border-destructive",
  warning: "border-warning",
};

export function UploadStep({ icon: Icon, title, description, status }: UploadStepProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`flex items-center gap-4 p-4 rounded-2xl border-2 bg-card ${statusBorderColors[status]}`}
    >
      <div className={`p-3 rounded-xl ${statusColors[status]}`}>
        <Icon className="w-6 h-6" />
      </div>
      <div className="flex-1">
        <h4 className="font-semibold text-foreground">{title}</h4>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      {status === "active" && (
        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      )}
    </motion.div>
  );
}
