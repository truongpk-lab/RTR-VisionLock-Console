import { useLocation, useOutlet } from "react-router";
import { AnimatePresence, motion } from "motion/react";

// Animates route changes inside the main content area. useOutlet() snapshots the
// current route element so the outgoing page keeps rendering through its exit
// animation while the incoming page fades/slides in.
export function PageTransition() {
  const location = useLocation();
  const outlet = useOutlet();

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={location.pathname}
        className="absolute inset-0"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
      >
        {outlet}
      </motion.div>
    </AnimatePresence>
  );
}
