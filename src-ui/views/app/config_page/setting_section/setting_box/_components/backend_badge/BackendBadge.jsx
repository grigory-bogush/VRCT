import clsx from "clsx";
import { useI18n } from "@useI18n";
import GpuSvg from "@images/gpu.svg?react";
import styles from "./BackendBadge.module.scss";

// Where a local model / compute device runs, so it is obvious which models
// work on an AMD (or Intel) graphics card:
//   "vulkan"   -> AMD / Intel GPU (llama.cpp / whisper.cpp through Vulkan)
//   "cpu_cuda" -> CPU / NVIDIA (CTranslate2, faster-whisper)
//   "cuda"     -> NVIDIA GPU (a CUDA compute device)
//   "cpu"      -> CPU (the CPU compute device)
export const BackendBadge = ({ backend, compact = false }) => {
    const { t } = useI18n();
    if (!backend) return null;

    const is_gpu = backend === "vulkan" || backend === "cuda";
    const label = t(`config_page.common.backend_badge.${backend}`);
    let title;
    if (backend === "vulkan") title = t("config_page.common.backend_badge.vulkan_title");
    if (backend === "cpu_cuda") title = t("config_page.common.backend_badge.cpu_cuda_title");

    return (
        <span
            className={clsx(styles.badge, styles[backend], { [styles.compact]: compact })}
            title={title}
        >
            {is_gpu && <GpuSvg className={styles.gpu_svg} />}
            <span className={styles.badge_label}>{label}</span>
        </span>
    );
};

// Backend of a compute-device entry from the sidecar
// ({device: "cpu" | "cuda" | "vulkan", ...}).
export const deviceBackend = (device) => {
    switch (device?.device) {
        case "vulkan": return "vulkan";
        case "cuda": return "cuda";
        case "cpu": return "cpu";
        default: return null;
    }
};

// Compute-device dropdown label: device name plus a backend badge. The
// sidecar names Vulkan devices "<GPU> (Vulkan)"; the badge says it instead.
export const DeviceLabel = ({ device, label }) => {
    const backend = deviceBackend(device);
    const text = backend === "vulkan" ? String(label).replace(/ \(Vulkan\)(?=(:\d+)?$)/, "") : label;
    return (
        <span className={styles.device_label}>
            <span className={styles.device_name}>{text}</span>
            <BackendBadge backend={backend} compact={true} />
        </span>
    );
};
