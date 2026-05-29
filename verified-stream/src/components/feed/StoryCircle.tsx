import { motion } from "framer-motion";

interface StoryCircleProps {
  image: string;
  username: string;
  isLive?: boolean;
  hasStory?: boolean;
}

export function StoryCircle({ image, username, isLive, hasStory = true }: StoryCircleProps) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className="flex flex-col items-center gap-1 cursor-pointer"
    >
      <div className={`relative p-[3px] rounded-full ${hasStory ? 'gradient-primary' : 'bg-border'}`}>
        <div className="bg-background p-[2px] rounded-full">
          <img
            src={image}
            alt={username}
            className="w-16 h-16 rounded-full object-cover"
          />
        </div>
        {isLive && (
          <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 px-2 py-0.5 text-[10px] font-semibold text-primary-foreground gradient-primary rounded-md">
            LIVE
          </span>
        )}
      </div>
      <span className="text-xs text-muted-foreground truncate max-w-16">
        {username}
      </span>
    </motion.div>
  );
}
