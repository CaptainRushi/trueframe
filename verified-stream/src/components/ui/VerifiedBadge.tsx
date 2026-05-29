import { ShieldCheck } from "lucide-react";
import { motion } from "framer-motion";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface VerifiedBadgeProps {
  size?: "sm" | "md" | "lg";
  showTooltip?: boolean;
  timestamp?: string;
}

const sizeClasses = {
  sm: "w-4 h-4",
  md: "w-5 h-5",
  lg: "w-6 h-6",
};

const containerClasses = {
  sm: "p-0.5",
  md: "p-1",
  lg: "p-1.5",
};

export function VerifiedBadge({ size = "md", showTooltip = true, timestamp }: VerifiedBadgeProps) {
  const badge = (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      className={`inline-flex items-center justify-center rounded-full bg-verified ${containerClasses[size]} shadow-verified`}
    >
      <ShieldCheck className={`${sizeClasses[size]} text-verified-foreground`} />
    </motion.div>
  );

  if (!showTooltip) return badge;

  return (
    <Tooltip>
      <TooltipTrigger asChild>{badge}</TooltipTrigger>
      <TooltipContent className="bg-verified text-verified-foreground border-verified">
        <p className="font-medium">Verified Real Content</p>
        {timestamp && (
          <p className="text-xs opacity-80">Verified {timestamp}</p>
        )}
      </TooltipContent>
    </Tooltip>
  );
}
