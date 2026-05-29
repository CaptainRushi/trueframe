import { useEffect } from 'react';

/**
 * Hook to automatically reload the page if a new version is detected.
 * This is useful for Vercel deployments where the browser might cache old assets.
 */
export const useAutoReload = () => {
    useEffect(() => {
        // Only run in production
        if (import.meta.env.DEV) return;

        const CHECK_INTERVAL = 1000 * 60 * 5; // Check every 5 minutes

        const checkVersion = async () => {
            try {
                // We look for a version.json file in the public folder
                // Which should be generated during the build process
                const response = await fetch('/version.json', { cache: 'no-store' });
                if (!response.ok) return;

                const data = await response.json();
                const latestVersion = data.version;
                const currentVersion = localStorage.getItem('app_version');

                if (!currentVersion) {
                    localStorage.setItem('app_version', latestVersion);
                    return;
                }

                if (currentVersion !== latestVersion) {
                    console.log('New version detected, reloading...');
                    localStorage.setItem('app_version', latestVersion);

                    // Small delay to let user see status if we had a UI for it
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                }
            } catch (error) {
                console.error('Failed to check for updates:', error);
            }
        };

        const interval = setInterval(checkVersion, CHECK_INTERVAL);
        checkVersion(); // Initial check

        return () => clearInterval(interval);
    }, []);
};
