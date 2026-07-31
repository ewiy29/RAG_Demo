import { styled } from "@mui/material/styles";
import ListItem from "@mui/material/ListItem";

/** Document row that dims while its delete request is in flight. */
export const DocumentRow = styled(ListItem, {
  shouldForwardProp: (prop) => prop !== "$deleting",
})<{ $deleting: boolean }>(({ $deleting }) => ({
  opacity: $deleting ? 0.5 : 1,
}));
