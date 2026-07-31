import { useState, type MouseEvent } from "react";
import Divider from "@mui/material/Divider";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import AddCircleOutlinedIcon from "@mui/icons-material/AddCircleOutlined";
import CheckIcon from "@mui/icons-material/Check";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";

import { useUser } from "../context/UserContext";
import { ShortId, SwitcherButton, SwitcherLabel } from "./UserSwitcher.styled";

interface UserSwitcherProps {
  activeUserId: string;
  onSwitch: (id: string) => void;
  onAddUser: () => void;
}

const shortId = (id: string) => id.slice(0, 8);
const labelFor = (index: number) => `User ${index + 1}`;

/**
 * Header control for the demo's tenant roster: shows the active user and lets
 * you switch between remembered tenant GUIDs (or mint a new one) to showcase
 * per-user document isolation. Not authentication -- see the README.
 */
export function UserSwitcher({
  activeUserId,
  onSwitch,
  onAddUser,
}: UserSwitcherProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const open = Boolean(anchorEl);
  // Roster comes from context; a change to the active id re-renders this so the
  // roster (including a just-added user) stays in sync with the active id.
  const { roster } = useUser();
  const activeIndex = roster.indexOf(activeUserId);

  const openMenu = (e: MouseEvent<HTMLElement>) => setAnchorEl(e.currentTarget);
  const closeMenu = () => setAnchorEl(null);

  const handleSwitch = (id: string) => {
    closeMenu();
    if (id !== activeUserId) {
      onSwitch(id);
    }
  };

  const handleAdd = () => {
    closeMenu();
    onAddUser();
  };

  return (
    <>
      <SwitcherButton
        color="inherit"
        size="small"
        onClick={openMenu}
        startIcon={<PersonOutlinedIcon />}
        aria-haspopup="menu"
        aria-expanded={open || undefined}
        aria-label="Switch user"
      >
        <SwitcherLabel>
          <span>{activeIndex >= 0 ? labelFor(activeIndex) : "User"}</span>
          <ShortId>{shortId(activeUserId)}</ShortId>
        </SwitcherLabel>
      </SwitcherButton>

      <Menu anchorEl={anchorEl} open={open} onClose={closeMenu}>
        {roster.map((id, index) => (
          <MenuItem
            key={id}
            selected={id === activeUserId}
            onClick={() => handleSwitch(id)}
          >
            <ListItemIcon>
              {id === activeUserId ? <CheckIcon fontSize="small" /> : null}
            </ListItemIcon>
            <ListItemText primary={labelFor(index)} secondary={shortId(id)} />
          </MenuItem>
        ))}
        <Divider />
        <MenuItem onClick={handleAdd}>
          <ListItemIcon>
            <AddCircleOutlinedIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="New user" />
        </MenuItem>
      </Menu>
    </>
  );
}
