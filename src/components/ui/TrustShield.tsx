import { ShieldCheck, ShieldAlert, ShieldX } from "lucide-react";
import { motion } from "framer-motion";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface TrustShieldProps {
  trustScore: number;
  status: string;
  size?: "sm" | "md" | "lg";
  showScore?: boolean;
  showTooltip?: boolean;
}

const sizeConfig = {
  sm: { icon: "w-4 h-4", container: "p-0.5", text: "text-xs" },
  md: { icon: "w-5 h-5", container: "p-1", text: "text-sm" },
  lg: { icon: "w-6 h-6", container: "p-1.5", text: "text-base" },
};

function getShieldConfig(trustScore: number, status: string) {
  if (status === "RESTRICTED" || trustScore < 25) {
    return {
      Icon: ShieldX,
      color: "text-destructive",
      bg: "bg-destructive/10",
      border: "border-destructive/20",
      label: "Restricted",
      tooltipBg: "bg-destructive",
    };
  }
  if (status === "AT_RISK" || (trustScore >= 25 && trustScore < 50)) {
    return {
      Icon: ShieldAlert,
      color: "text-warning",
      bg: "bg-warning/10",
      border: "border-warning/20",
      label: "At Risk",
      tooltipBg: "bg-warning",
    };
  }
  if (status === "NEW_USER") {
    return {
      Icon: ShieldAlert,
      color: "text-blue-400",
      bg: "bg-blue-400/10",
      border: "border-blue-400/20",
      label: "New User",
      tooltipBg: "bg-blue-500",
    };
  }
  return {
    Icon: ShieldCheck,
    color: "text-verified",
    bg: "bg-verified/10",
    border: "border-verified/20",
    label: "Trusted",
    tooltipBg: "bg-verified",
  };
}

export function TrustShield({
  trustScore,
  status,
  size = "md",
  showScore = false,
  showTooltip = true,
}: TrustShieldProps) {
  const config = getShieldConfig(trustScore, status);
  const sizes = sizeConfig[size];

  const badge = (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      className={`inline-flex items-center gap-1 rounded-full ${config.bg} ${sizes.container} border ${config.border}`}
    >
      <config.Icon className={`${sizes.icon} ${config.color}`} />
      {showScore && (
        <span className={`${sizes.text} font-bold ${config.color} pr-1`}>
          {trustScore}
        </span>
      )}
    </motion.div>
  );

  if (!showTooltip) return badge;

  return (
    <Tooltip>
      <TooltipTrigger asChild>{badge}</TooltipTrigger>
      <TooltipContent className={`${config.tooltipBg} text-white border-none`}>
        <p className="font-medium">{config.label} Creator</p>
        <p className="text-xs opacity-80">Trust Score: {trustScore}/100</p>
      </TooltipContent>
    </Tooltip>
  );
}
