import { ShieldCheck, Camera, AlertTriangle, ShieldAlert, Edit3 } from "lucide-react";
import { motion } from "framer-motion";

interface AuthenticityLabelProps {
  label: string;
}

const labelConfig: Record<string, { icon: typeof ShieldCheck; text: string; bg: string; textColor: string }> = {
  CAMERA_ORIGINAL: {
    icon: Camera,
    text: "Captured on TrueFrame",
    bg: "bg-green-600/90",
    textColor: "text-white",
  },
  VERIFIED_REAL: {
    icon: ShieldCheck,
    text: "Verified Real",
    bg: "bg-verified/90",
    textColor: "text-verified-foreground",
  },
  EDITED: {
    icon: Edit3,
    text: "Minor Edits",
    bg: "bg-yellow-500/90",
    textColor: "text-black",
  },
  REJECTED_SYNTHETIC: {
    icon: ShieldAlert,
    text: "Suspicious",
    bg: "bg-red-500/90",
    textColor: "text-white",
  },
  REJECTED_MISLEADING: {
    icon: AlertTriangle,
    text: "Misleading",
    bg: "bg-orange-500/90",
    textColor: "text-white",
  },
};

export function AuthenticityLabel({ label }: AuthenticityLabelProps) {
  const config = labelConfig[label] || labelConfig.VERIFIED_REAL;
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`flex items-center gap-1.5 ${config.bg} backdrop-blur-sm px-3 py-1.5 rounded-full`}
    >
      <Icon className={`w-3.5 h-3.5 ${config.textColor}`} />
      <span className={`text-xs font-medium ${config.textColor}`}>
        {config.text}
      </span>
    </motion.div>
  );
}
